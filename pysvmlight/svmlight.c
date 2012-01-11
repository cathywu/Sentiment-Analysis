#include <Python.h>
#include "svm_common.h"
#include "svm_learn.h"

/* This function corresponds to parse_document() on line 763 of svm_common.c, but
 * it extracts data from a PyObject instead of a string. In Python, an individual
 * document should look like this:
 * >> (<label>, [(<feature>, <value>), ...]) */
static int unpack_document(
        PyObject *docobj, WORD *words, double *label,
        long *queryid, long *slackid, double *costfactor,
        long int *numwords, long int max_words_doc);

/* Count certain properties of a doclist:
 *  - max_docs is simply the length of the provided list.
 *  - max_words is the maximum number of words in any of the documents. */
static void count_doclist(
        PyObject *doclist, long *max_docs, long *max_words);

/* Unpack a list containing documents (training examples) and labels. */
static int unpack_doclist(
        PyObject *doclist, DOC ***docs, double **label, int *totwords, int *totdoc);

/* This auxiliary function to svm_learn reads some parameters from the keywords to
 * the function and fills the rest in with defaults (from read_input_parameters()
 * in svm_learn_main.c:109). */
static int read_learning_parameters(
        PyObject *kwds, long *verbosity, LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm);

/* The only thing we need in this whole struct is the MODEL, but since the MODEL
 * apparently depends on the list of training examples, we have to keep track of
 * that too. Just use the macro GET_MODEL(). */
typedef struct {
    MODEL *model;
    DOC **docs;
    int totdoc;
} MODEL_AND_DOCS;

/* Get the MODEL out of a (void) pointer to a MODEL_AND_DOCS. */
#define GET_MODEL(ptr) ((MODEL_AND_DOCS *)ptr)->model

/* Destructor function for MODEL_AND_DOCS. */
static void free_model_and_docs(void *ptr);

static void free_just_model(void *ptr);

static PyObject *svm_learn(PyObject *self, PyObject *args, PyObject *kwds);
static PyObject *py_write_model(PyObject *self, PyObject *args);
static PyObject *py_read_model(PyObject *self, PyObject *args);
static PyObject *svm_classify(PyObject *self, PyObject *args);

static PyMethodDef PySVMLightMethods[] = {
    {"learn", svm_learn, METH_VARARGS | METH_KEYWORDS,
     "learn(training_data, **options) -> model"},
    {"classify", svm_classify, METH_VARARGS,
     "classify(model, test_data, **options) -> predictions"},
    {"write_model", py_write_model, METH_VARARGS,
     "write_model(model, filename) -> None"},
    {"read_model", py_read_model, METH_VARARGS,
     "read_model(filename) -> model"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initsvmlight(void)
{
    PyObject *module;
    module = Py_InitModule("svmlight", PySVMLightMethods);
    if (module == NULL)
        return;
}

/* ========================================================================== */

static int unpack_document(
        PyObject *docobj, WORD *words, double *label,
        long *queryid, long *slackid, double *costfactor,
        long int *numwords, long int max_words_doc)
{
    long wpos = 0;
    double weight;
    double wnum_d; /* We read wnum as a float instead of an integer to avoid a
                    * deprecation warning (since it doesn't matter in Python). */
    PyObject *iter, *featurepair, *words_list;

    /* We initialize these parameters with their default values, since we won't
     * be reading them from the feature pairs (don't really care). */
    (*queryid) = 0;
    (*slackid) = 0;
    (*costfactor) = 1;

    if(!PyTuple_Check(docobj)) {
        PyErr_SetString(PyExc_TypeError, "document should be a tuple");
        return 0;
    }
    if(!PyArg_ParseTuple(docobj, "dO|l", label, &words_list, queryid)) 
        return 0;
    if(!PyList_Check(words_list)) {
        PyErr_SetString(PyExc_TypeError, "expected list of feature pairs");
        return 0;
    }
    iter = PyObject_GetIter(words_list);
    while((featurepair = PyIter_Next(iter)) && wpos < max_words_doc) {
        if(!PyArg_ParseTuple(featurepair, "dd", &wnum_d, &weight)) {
            Py_DECREF(iter);
            Py_DECREF(featurepair);
            return 0;
        }
        words[wpos].wnum = (long)wnum_d;
        words[wpos].weight = (FVAL)weight;
        wpos++;
        Py_DECREF(featurepair);
    }
    Py_DECREF(iter);
    words[wpos].wnum = 0; /* Sentinel value */
    (*numwords) = wpos + 1;
    return 1;
}

static void count_doclist(
        PyObject *doclist, long *max_docs, long *max_words)
{
    PyObject *iter, *item;

    if(max_docs != NULL)
        *max_docs = PySequence_Length(doclist);

    if(max_words != NULL) {
        *max_words = 0;
        iter = PyObject_GetIter(doclist);
        while((item = PyIter_Next(iter))) {
            PyObject *words_list;
            Py_ssize_t len;
            if(PyTuple_Check(item)) {
                if((words_list = PyTuple_GetItem(item, 1))) {
                    if(PySequence_Check(words_list)) {
                        len = PySequence_Length(words_list);
                        if(len > *max_words) *max_words = len;
                    }
                } else
                    PyErr_Clear();
            }
            Py_DECREF(item);
        }
        Py_DECREF(iter);
    }
}

static int unpack_doclist(
        PyObject *doclist, DOC ***docs, double **label, int *totwords, int *totdoc)
{
    long queryid, slackid, dnum = 0, wpos, max_docs, max_words;
    WORD *words;
    double doc_label, costfactor;
    PyObject *iter, *item;

    if(!PySequence_Check(doclist)) {
        PyErr_SetString(PyExc_TypeError, "expected list of documents");
        return 0;
    }
    count_doclist(doclist, &max_docs, &max_words);
    (*docs) = (DOC **)malloc(sizeof(DOC*) * max_docs); /* Feature vectors */
    (*label) = (double *)malloc(sizeof(double) * max_docs); /* Target values */
    words = (WORD *)malloc(sizeof(WORD) * (max_words + 1));

    (*totwords) = 0;
    iter = PyObject_GetIter(doclist);
    while((item = PyIter_Next(iter))) {
        if(!unpack_document(item, words, &doc_label, &queryid, &slackid,
                             &costfactor, &wpos, max_words))
            return 0;
        if((wpos > 1) && ((words[wpos - 2]).wnum > (*totwords)))
            (*totwords)=(words[wpos-2]).wnum;
        (*label)[dnum] = doc_label;
        (*docs)[dnum] = create_example(dnum, queryid, slackid, costfactor,
                                       create_svector(words, "", 1.0));
        dnum++;
        Py_DECREF(item);
    }
    Py_DECREF(iter);

    free(words);
    (*totdoc) = dnum;
    return 1;
}

static int read_learning_parameters(
        PyObject *kwds, long *verbosity, LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm)
{
    strcpy(learn_parm->predfile, "trans_predictions");
    strcpy(learn_parm->alphafile, "");
    (*verbosity) = 0;
    learn_parm->biased_hyperplane = 1;
    learn_parm->sharedslack = 0;
    learn_parm->remove_inconsistent = 0;
    learn_parm->skip_final_opt_check = 0;
    learn_parm->svm_maxqpsize = 10;
    learn_parm->svm_newvarsinqp = 0;
    learn_parm->svm_iter_to_shrink = -9999;
    learn_parm->maxiter = 100000;
    learn_parm->kernel_cache_size = 40;
    learn_parm->svm_c = 0.0;
    learn_parm->eps = 0.1;
    learn_parm->transduction_posratio = -1.0;
    learn_parm->svm_costratio = 1.0;
    learn_parm->svm_costratio_unlab = 1.0;
    learn_parm->svm_unlabbound = 1E-5;
    learn_parm->epsilon_crit = 0.001;
    learn_parm->epsilon_a = 1E-15;
    learn_parm->compute_loo = 0;
    learn_parm->rho = 1.0;
    learn_parm->xa_depth = 0;
    kernel_parm->kernel_type = 0;
    kernel_parm->poly_degree = 3;
    kernel_parm->rbf_gamma = 1.0;
    kernel_parm->coef_lin = 1;
    kernel_parm->coef_const = 1;
    strcpy(kernel_parm->custom,"empty");
    learn_parm->type = CLASSIFICATION;

    if(PyMapping_HasKeyString(kwds, "type")) {
        char *type = PyString_AsString(PyMapping_GetItemString(kwds, "type"));
        if(!type) return 0;
        else if(!strcmp(type, "classification")) learn_parm->type = CLASSIFICATION;
        else if(!strcmp(type, "regression")) learn_parm->type = REGRESSION;
        else if(!strcmp(type, "ranking")) learn_parm->type = RANKING;
        else if(!strcmp(type, "optimization")) learn_parm->type = OPTIMIZATION;
        else {
            PyErr_SetString(PyExc_ValueError, "unknown learning type specified. Valid types are: 'classification', 'regression', 'ranking' and 'optimization'.");
            return 0;
        }
    }
    if(PyMapping_HasKeyString(kwds, "kernel")) {
        char *kernel = PyString_AsString(PyMapping_GetItemString(kwds, "kernel"));
        if(!kernel) return 0;
        else if(!strcmp(kernel, "linear")) kernel_parm->kernel_type = LINEAR;
        else if(!strcmp(kernel, "polynomial")) kernel_parm->kernel_type = POLY;
        else if(!strcmp(kernel, "rbf")) kernel_parm->kernel_type = RBF;
        else if(!strcmp(kernel, "sigmoid")) kernel_parm->kernel_type = SIGMOID;
        else {
            PyErr_SetString(PyExc_ValueError, "unknown kernel type specified. Valid types are: 'linear', 'polynomial', 'rbf' and 'sigmoid'.");
            return 0;
        }
    }
    if(PyMapping_HasKeyString(kwds, "verbosity")) {
        PyObject *vobj = PyMapping_GetItemString(kwds, "verbosity");
        (*verbosity) = PyNumber_AsSsize_t(vobj, 0);
    }
    if(PyMapping_HasKeyString(kwds, "C")) {
        PyObject *vobj = PyMapping_GetItemString(kwds, "C");
        learn_parm->svm_c = PyFloat_AsDouble(vobj);
    }
    if(PyMapping_HasKeyString(kwds, "poly_degree")) {
        PyObject *vobj = PyMapping_GetItemString(kwds, "poly_degree");
        kernel_parm->poly_degree = PyNumber_AsSsize_t(vobj, 0);
    }
    if(PyMapping_HasKeyString(kwds, "rbf_gamma")) {
        PyObject *vobj = PyMapping_GetItemString(kwds, "rbf_gamma");
        kernel_parm->rbf_gamma = PyFloat_AsDouble(vobj);
    }
    if(PyMapping_HasKeyString(kwds, "coef_lin")) {
        PyObject *vobj = PyMapping_GetItemString(kwds, "coef_lin");
        kernel_parm->coef_lin = PyFloat_AsDouble(vobj);
    }
    if(PyMapping_HasKeyString(kwds, "coef_const")) {
        PyObject *vobj = PyMapping_GetItemString(kwds, "coef_const");
        kernel_parm->coef_const = PyFloat_AsDouble(vobj);
    }

    if(learn_parm->svm_iter_to_shrink == -9999) {
        if(kernel_parm->kernel_type == LINEAR)
            learn_parm->svm_iter_to_shrink=2;
        else
            learn_parm->svm_iter_to_shrink=100;
    }

    return 1;
}

void free_model_and_docs(void *ptr) {
    int i;
    MODEL_AND_DOCS *obj = (MODEL_AND_DOCS *)ptr;
    free_model(obj->model, 0);
    for(i = 0; i < obj->totdoc; i++)
        free_example(obj->docs[i], 1);
    free(obj->docs);
    free(ptr);
}

void free_just_model(void *ptr) {
    free_model(GET_MODEL(ptr), 1);
    free(ptr);
}

static PyObject *svm_learn(PyObject *self, PyObject *args, PyObject *kwds)
{
    DOC **docs;
    double* target;
    int totwords, totdoc;
    KERNEL_CACHE *kernel_cache;
    LEARN_PARM learn_parm;
    KERNEL_PARM kernel_parm;
    long verbosity;
    PyObject *doclist;
    MODEL *model;
    MODEL_AND_DOCS *result;

    if(!PyArg_ParseTuple(args, "O", &doclist))
        return NULL;
    read_learning_parameters(kwds, &verbosity, &learn_parm, &kernel_parm);
    if(!unpack_doclist(doclist, &docs, &target, &totwords, &totdoc))
        return NULL;

    model = malloc(sizeof(MODEL));
    if(kernel_parm.kernel_type == LINEAR)
        kernel_cache = NULL;
    else
        kernel_cache = kernel_cache_init(totdoc, learn_parm.kernel_cache_size);

    if(learn_parm.type == CLASSIFICATION) {
        svm_learn_classification(docs, target, totdoc, totwords, &learn_parm,
                                 &kernel_parm, kernel_cache, model, NULL /* alpha_in */);
    }
    else if(learn_parm.type == REGRESSION) {
        svm_learn_regression(docs, target, totdoc, totwords, &learn_parm,
                             &kernel_parm, &kernel_cache, model);
    }
    else if(learn_parm.type == RANKING) {
        svm_learn_ranking(docs, target, totdoc, totwords, &learn_parm,
                          &kernel_parm, &kernel_cache, model);
    }
    else if(learn_parm.type == OPTIMIZATION) {
        svm_learn_optimization(docs, target, totdoc, totwords, &learn_parm,
                               &kernel_parm, kernel_cache, model, NULL /* alpha_in */);
    }

    /* Cleanup */
    if(kernel_cache)
        kernel_cache_cleanup(kernel_cache);
    free(target);

    result = (MODEL_AND_DOCS *)malloc(sizeof(MODEL_AND_DOCS));
    result->model = model;
    result->docs = docs;
    result->totdoc = totdoc;
    return PyCObject_FromVoidPtr(result, free_model_and_docs);
}

static PyObject *py_write_model(PyObject *self, PyObject *args) {
    char *modelfile;
    PyObject *modelobj;
    MODEL *model;

    if(!PyArg_ParseTuple(args, "Os", &modelobj, &modelfile))
        return NULL;
    model = GET_MODEL(PyCObject_AsVoidPtr(modelobj));
    write_model(modelfile, model);

    Py_RETURN_NONE;
}

static PyObject *py_read_model(PyObject *self, PyObject *args) {
    char *modelfile;
    MODEL *model;
    MODEL_AND_DOCS *result;

    if(!PyArg_ParseTuple(args, "s", &modelfile))
        return NULL;

    model = read_model(modelfile);
    result = (MODEL_AND_DOCS *)malloc(sizeof(MODEL_AND_DOCS));
    result->model = model;
    result->docs = 0;
    result->totdoc = 0;
    return PyCObject_FromVoidPtr(result, free_just_model);
}

static PyObject *svm_classify(PyObject *self, PyObject *args) {
    MODEL *model;
    PyObject *modelobj, *doclist, *iter, *item, *result;
    long max_words, max_doc;
    int docnum = 0, j;
    double dist;

    DOC *doc;
    WORD *words;
    long queryid, slackid, wnum;
    double costfactor, doc_label;


    if(!PyArg_ParseTuple(args, "OO", &modelobj, &doclist))
        return NULL;
    if(!PyCObject_Check(modelobj) || !PySequence_Check(doclist)) {
        PyErr_SetString(PyExc_TypeError, "invalid positional argument(s)");
        return NULL;
    }
    model = GET_MODEL(PyCObject_AsVoidPtr(modelobj));

    if(model->kernel_parm.kernel_type == 0)
        add_weight_vector_to_linear_model(model);

    count_doclist(doclist, &max_doc, &max_words);
    words = (WORD *)malloc(sizeof(WORD) * (max_words + 10));
    result = PyList_New(max_doc);

    iter = PyObject_GetIter(doclist);
    while((item = PyIter_Next(iter))) {
        unpack_document(item, words, &doc_label, &queryid, &slackid, &costfactor,
                        &wnum, max_words);
        Py_DECREF(item);
        if(model->kernel_parm.kernel_type == 0) { /* Linear kernel */
            /* "Check if feature numbers are not larger than in model. Remove
             * feature if necessary" -- svm_classify.c:77 */
            for(j= 0; (words[j]).wnum != 0; j++) {
                if(words[j].wnum > model->totwords)
                    words[j].wnum = 0;
            }
            doc = create_example(-1, 0, 0, 0.0, create_svector(words, "", 1.0));
            dist = classify_example_linear(model, doc);
            free_example(doc, 1);
        } else {
            doc = create_example(-1, 0, 0, 0.0, create_svector(words, "", 1.0));
            dist = classify_example(model, doc);
            free_example(doc, 1);
        }
        PyList_SetItem(result, docnum, PyFloat_FromDouble(dist));
        docnum++;
    }
    Py_DECREF(iter);
    free(words);
    return result;
}
