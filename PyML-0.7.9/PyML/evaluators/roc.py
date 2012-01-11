"""
Module for computing ROC curves and AUC scores

Theoretical and pratical concepts from 
Fawcett, T.  ROC graphs: Notes and pratical considerations
for data mining researchers.  HPL-2003-4, 2003.

By Michael Hamilton, 2009
With edits by Asa Ben-Hur 2010
"""

import numpy
from bisect import bisect_left # for binary search

__docformat__ = "restructuredtext en"

def roc( dvals, labels, rocN=None, normalize=True ) :
    """
    Compute ROC curve coordinates and area

    - `dvals`  - a list with the decision values of the classifier
    - `labels` - list with class labels, \in {0, 1} 

    returns (FP coordinates, TP coordinates, AUC )
    """
    if rocN is not None and rocN < 1 :
        rocN = int(rocN * numpy.sum(numpy.not_equal(labels, 1)))

    TP = 0.0  # current number of true positives
    FP = 0.0  # current number of false positives
    
    fpc = [ 0.0 ]  # fp coordinates
    tpc = [ 0.0 ]  # tp coordinates
    dv_prev = -numpy.inf # previous decision value
    TP_prev = 0.0
    FP_prev = 0.0
    area = 0.0

    num_pos = labels.count( 1 )  # number of pos labels
    num_neg = labels.count( 0 ) # number of neg labels
    
    if num_pos == 0 or num_pos == len(labels) :
        raise ValueError, "There must be at least one example from each class"

    # sort decision values from highest to lowest
    indices = numpy.argsort( dvals )[ ::-1 ]
    
    for idx in indices:
        # increment associated TP/FP count
        if labels[ idx ] == 1:
            TP += 1.
        else:
            FP += 1.
            if rocN is not None and FP == rocN : 
                break
        # Average points with common decision values
        # by not adding a coordinate until all
        # have been processed
        if dvals[ idx ] != dv_prev:
            if len(fpc) > 0 and FP == fpc[-1] :
                tpc[-1] = TP
            else :
                fpc.append( FP  )
                tpc.append( TP  )
            dv_prev = dvals[ idx ]
            area += _trap_area( ( FP_prev, TP_prev ), ( FP, TP ) )
            FP_prev = FP
            TP_prev = TP

    #area += _trap_area( ( FP, TP ), ( FP_prev, TP_prev ) )
    #fpc.append( FP  )
    #tpc.append( TP )
    if normalize :
        fpc = [ float( x ) / FP for x in fpc ]
        if TP > 0:
            tpc = [ float( x ) / TP for x in tpc ]
        if area > 0:
            area /= ( TP * FP )

    return fpc, tpc, area
    
def roc_VA( folds, rocN=None, n_samps=100 ):
    """
    Compute ROC curve using vertical averaging

    `folds` - list of ( labels, dvals ) pairs where labels
              is a list of class labels and dvals are
              the decision values of the classifier
    """
    # return variables
    invl = 1.0 / n_samps # interval to sample FPR
    FPRs = numpy.arange( 0, (1+invl), invl )
    TPRs = [ ] # will contain assoc TPR avgs for FPRs
    # folds must be listified
    assert type( folds ) == type( [ ] )
    rocs = [ ] # list of roc tuples ( [FPR,TPR] ) for folds
    areas = [ ] # individual AUCs for each fold

    # calculate individual ROC curves for each fold
    for dvals,labels in folds:
        fpc, tpc, area = roc( dvals, labels, rocN )
        rocs.append( (fpc, tpc) )
        areas.append( area )

    for fpr in FPRs:
        # accumulate TPRs for current fpr over all folds
        tpr_folds = [ ] 
        # fix FPR and accumulate (interpolated) TPRs
        for fpc, tpc in rocs:
            tpr_folds.append( _tpr_for_fpr( fpc, tpc, fpr ))
        # average tprs and append
        TPRs.append( numpy.mean( tpr_folds ) )
    
    return FPRs, numpy.array( TPRs ), numpy.mean( areas )


def _tpr_for_fpr( fpc, tpc, fpr ):
    """
    Returns the (estimated) tpr for the given fpr for 
    the given false positive/true positive coordinates
    from an ROC curve.

    `fpc` - False positive coordinates from ROC curve
    `tpc` - True positive coordinates from ROC curve
    """
    
    # take advantage of monotonic property of ROC curves
    # and search for fpr in O( log n ) time
    idx = bisect_left( fpc, fpr, 0, len(fpc)-1 )
    
    # if exact match, then return
    if fpc[ idx ] == fpr:
        return tpc[ idx ]

    else:
        # check if idx is last index of fpc
        #if idx == len( fpc ) - 1:
        #    return tpc[ idx ]

        # check if the neighboring fprs are identical
        #elif fpc[ idx ] == fpc[ idx + 1 ]:
            # return the average of the tprs 
        #    return ( tpc[ idx ] + tpc[ idx+1 ] ) / 2.0
        # otherwise, interpolate the tpr
        return _interpolate( ( fpc[ idx-1 ], tpc[ idx-1 ] ),
                             ( fpc[ idx ], tpc[ idx ] ),
                             fpr )

def _interpolate( p1, p2, x ):
    """
    Interpolate the value of f( x ).

    `p1` - 1st interpolation point (x1, y1)
    `p1` - 2nd interpolation point (x2, y2)
    `x`  - the value to interpolate
    """
    return p1[ 1 ] + _slope( p1, p2 ) * ( x - p1[ 0 ] )

def _trap_area( p1, p2 ):
    """
    Calculate the area of the trapezoid defined by points
    p1 and p2
    
    `p1` - left side of the trapezoid
    `p2` - right side of the trapezoid
    """
    base = abs( p2[ 0 ] - p1[ 0 ] )
    avg_ht = ( p1[ 1 ] + p2[ 1 ] ) / 2.0

    return base * avg_ht

def _slope( p1, p2 ):
    """
    Calculates the slope of the line defined by
    points p1 and p2
    """
    delta_x = p2[ 0 ] - p1[ 0 ]
    delta_y = p2[ 1 ] - p1[ 1 ]
    
    # if infinite slope, scream
    if delta_x == 0: raise( "Infinite slope" )

    return float( delta_y ) / delta_x

def plotROC(rocFP, rocTP, fileName = None, **args) :
    """plot the ROC curve from a given Results (or Results-like) object

    :Parameters:
      - `res` - Results (or Container object that was made by saving a a
        Results object (note that if you have a Results object you can
        use this function as a method so there is no need to supply this
        argument).
      - `fileName` - optional argument - if given, the roc curve is saved
        in the given file name.  The format is determined by the extension.
        Supported extensions: .eps, .png, .svg
    
    :Keywords:
      - `normalize` - whether to normalize the ROC curve (default: True)
      - `plotStr` - which string to pass to matplotlib's plot function
        default: 'ob'
      - `axis` - redefine the figure axes; takes a list of the form
        [xmin,xmax,ymin,ymax]
      - `show` - whether to show the ROC curve (default: True)
        useful when you just want to save the curve to a file.
        The use of Some file formats automatically sets this to False
        (e.g. svg files).  This relates to quirks of matplotlib.
    """

    if 'show' in args :
        show = args['show']
    else :
        show = True
    if 'plotStr' in args :
        plotStr = args['plotStr']
    else :
        plotStr = 'ob'
    rocNormalize = True
    if 'normalize' in args :
        rocNormalize = args['normalize']

    numPoints = 200
    if 'numPoints' in args :
        numPoints = args['numPoints']
        
    stride = int(max(1, float(len(rocTP)) / float(numPoints)))

    if stride > 1 :
        rocTP = [rocTP[i] for i in range(0,len(rocTP), stride)]
        rocFP = [rocFP[i] for i in range(0,len(rocFP), stride)]        
        
    import matplotlib
    if fileName is not None and fileName.find('.svg') > 0 :
        matplotlib.use('SVG')
        show = False
    if fileName is not None and fileName.find('.eps') > 0 :
        matplotlib.use('PS')
        show = False

    from matplotlib import pylab
    lines = pylab.plot(rocFP, rocTP, plotStr,
                        markersize = 8, linewidth = 3)
    if rocNormalize :
        pylab.xlabel('False positive rate', fontsize = 18)
        pylab.ylabel('True positive rate', fontsize = 18)
    else :
        pylab.xlabel('False positives', fontsize = 18)
        pylab.ylabel('True positives', fontsize = 18)
    if rocNormalize :
        pylab.axis([0, 1, 0, 1])
    if 'axis' in args :
        pylab.axis(args['axis'])
    if fileName is not None :
        pylab.savefig(fileName)
    if show :
        pylab.show()

def plotROCs(resList, descriptions = None, fileName = None, **args) :

    """
    plot multiple ROC curves.

    :Parameters:
      - `resList` - a list or dictionary of Result or Result-like objects
      - `descriptions` - text for the legend (a list the size of resList).
        A legend is not shown if this parameter is not given
        In the case of a dictionary input the description for the legend is
        taken from the dictionary keys.
      - `fileName` - if given, a file to save the figure in

    :Keywords:
      - `legendLoc` - the position of the legend -- an integer between 0 and 9;
        see the matplotlib documentation for details
      - `plotStrings` - a list of matlab style plotting string to send to the
        plotROC function (instead of the plotString keyword of plotROC)
      - `other keywords` - keywords of the plotROC function 
    """

    if type(resList) == type([]) and type(resList[0]) == type('') :
        fileNames = resList
        resList = []
        for fileName in fileNames :
            resList.append(myio.load(fileName))
        if descriptions is None :
            descriptions = []
            for fileName in fileNames :
                descriptions.append(os.path.splitext(fileName)[0])

    import matplotlib
    show = True
    if fileName is not None and fileName.find('.svg') > 0 :
        matplotlib.use('SVG')
        show = False
    if fileName is not None and fileName.find('.eps') > 0 :
        matplotlib.use('PS')
        show = False
                
    from matplotlib import pylab
    args['show'] = False
    
    plotStrings = ['bo', 'k^', 'rv', 'g<', 'm>', 'k<']
    plotStrings = ['b-', 'k--', 'r-', 'g-.', 'm-', 'k:', 'b-', 'r-', 'g-']
    #plotStrings = ['b:', 'k-.', 'b-', 'g-', 'm-', 'k-', 'b-', 'r-', 'g-']
    if 'plotStrings' in args :
        plotStrings = args['plotStrings']
    if type(resList) == type([]) :
        for i in range(len(resList)) :
            print i
            args['plotStr'] = plotStrings[i]
            plotROC(resList[i], **args)
    else :
        if descriptions is None :
            descriptions = [key for key in resList]
        i = 0
        for key in resList :
            args['plotStr'] = plotStrings[i]
            plotROC(resList[key], **args)
            i+=1

    if descriptions is not None :
        legendLoc = 'best'
        if 'legendLoc' in args :
            legendLoc = args['legendLoc']
        pylab.legend(descriptions, loc = legendLoc)
        leg = pylab.gca().get_legend()
        ltext  = leg.get_texts()  # all the text.Text instance in the legend
        llines = leg.get_lines()  # all the lines.Line2D instance in the legend
        frame  = leg.get_frame()  # the Rectangle instance surrounding the legend

        #frame.set_facecolor(0.80)     # set the frame face color to light gray
        for obj in ltext :
            obj.set_size(14)
        #leg.draw_frame(False)         # don't draw the legend frame

    if fileName is not None :
        pylab.savefig(fileName)
    if show :
        pylab.show()


def test():
    # Testing
    import numpy.random as rand
    import pylab

    ############## Make synthetic data #################

    # Simulate protein binding site by having few positive
    # class examples 
    folds = [ ]
    labs = [ -1 ] * 500
    labs.extend( [ 1 ] * 20 )
    N = 40  # number of sequences
    
    # set means of decision values
    mu_n = -1
    mu_p = 1
    
    # draw decision values from normal distribution
    # for each class
    for i in xrange( N ):
        preds = rand.normal( mu_n, 1, 500 ).tolist()
        preds.extend( rand.normal( mu_p, 1, 20 ).tolist() )
        folds.append( ( labs[ : ], preds ) )

    # Calculate average ROC curve
    fpr, tpr, tpr_ci, area, area_ci = roc_VA( folds )

    # Print AUC with confidence interval
    print "area: %f, 95%% CI: (%f, %f)" % ( area, area-area_ci, area+area_ci )

    ####### Plot ROC curve with 95% confidence band  #######
    fig = pylab.figure( )
    ax = fig.add_subplot( 111 )
    # fill upper band
    xs, ys = pylab.poly_between( fpr, tpr, numpy.array( tpr ) + numpy.array( tpr_ci ) ) 
    ax.fill( xs, ys, 'r' )
    # fill lower band
    xs, ys = pylab.poly_between( fpr, tpr, numpy.array( tpr ) - numpy.array( tpr_ci ) ) 
    ax.fill( xs, ys, 'r' )
    # label axes and main
    pylab.xlabel( "FPR" )
    pylab.ylabel( "TPR" )
    pylab.title( "Example ROC curve with 95% confidence band" )
    ax.plot( numpy.arange( 0, 1.1, .1 ), numpy.arange( 0, 1.1, .1  ), 'k--')
    pylab.xlim( -.01, 1.01 )
    pylab.ylim( -.01, 1.01 )
    pylab.show( )
    
