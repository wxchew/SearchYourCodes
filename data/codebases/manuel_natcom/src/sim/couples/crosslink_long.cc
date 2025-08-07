// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#include "crosslink_long.h"
#include "crosslink_prop.h"
#include "exceptions.h"
#include "random.h"
#include "modulo.h"
#include "meca.h"

extern Modulo const* modulo;
extern Random RNG;

//------------------------------------------------------------------------------

CrosslinkLong::CrosslinkLong(CrosslinkProp const* p, Vector const& w)
: Crosslink(p, w), mArm(nullTorque)
{
}


CrosslinkLong::~CrosslinkLong()
{
}

//------------------------------------------------------------------------------

#if ( DIM == 2 )

/**
 Returns -len or +len
 */
real CrosslinkLong::calcArm(const PointInterpolated & pt, Vector const& pos, real len)
{
    Vector vec = pt.pos() - pos;
    if ( modulo )
        modulo->fold(vec);
    return len * RNG.sign_exc( cross(vec, pt.diff()) );
}

#elif ( DIM == 3 )

/**
 Return a vector of norm `len`, perpendicular to the Fiber referenced by `pt` and aligned with the link.
 @todo update to match interSideLink3D when available
 */
Vector CrosslinkLong::calcArm(const PointInterpolated & pt, Vector const& pos, real len)
{
    Vector a  = pt.diff();
    Vector as = pos - pt.pos();
    if ( modulo )
        modulo->fold(as);
    Vector p = ( as - ( ( as * a ) / a.normSqr() ) * a );
    real pn = p.normSqr();
    if ( pn > REAL_EPSILON )
        return p * ( len / sqrt(pn) );
    else
        return a.randOrthoU(len);
    //return cross( pt.pos()-pos, pt.diff() ).normalized(len);
}

#endif

//------------------------------------------------------------------------------

Vector CrosslinkLong::posSide() const
{
#if ( DIM == 1 )
    
    return cHand1->pos();
    
#elif ( DIM == 2 )
    
    return cHand1->pos() + cross(mArm, cHand1->dirFiber());
    
#elif ( DIM == 3 )
    
    ///\todo: change formula to match interSideLink3D
    return cHand1->pos() + mArm;

#endif
}


/**
 Calculates the force for the interSideLink()
 */
Vector CrosslinkLong::force() const
{
    Vector d = cHand2->pos() - posSide();
    
    //correct for periodic space:
    if ( modulo )
        modulo->fold(d);
    
    return prop->stiffness * d;
}


/**
 This uses interSideLink().
 
 Another possibility would be SideSideLink, which is fully symmetric.
 */
void CrosslinkLong::setInteractions(Meca & meca) const
{
    PointInterpolated pt1 = cHand1->interpolation();
    PointInterpolated pt2 = cHand2->interpolation();
    
    /* 
     The 'arm' is recalculated each time, but in 2D at least,
     this maybe not necessary, as switching should be rare.
     */
    
#if ( DIM == 2 )
    
    mArm = calcArm(pt1, pt2.pos(), prop->length);
    meca.interSideLink2D(pt1, pt2, mArm, prop->stiffness);
    
#elif ( DIM == 3 )

    mArm = calcArm(pt1, pt2.pos(), prop->length);
    meca.interSideLinkS(pt1, pt2, mArm, prop->length, prop->stiffness);
    //@todo CrosslinkLong:setInteractions() use interSideLink3D()
    
#endif
    
    //meca.interSideSideLink( cHand1->interpolation(), cHand2->interpolation(), prop->length, prop->stiffness );
}


