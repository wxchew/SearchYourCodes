// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#include "trapper_long.h"
#include "trapper_prop.h"
#include "exceptions.h"
#include "random.h"
#include "modulo.h"
#include "meca.h"

extern Modulo const* modulo;
extern Random RNG;

//------------------------------------------------------------------------------

TrapperLong::TrapperLong(TrapperProp const* p, Vector const& w)
: Trapper(p, w), mArm(nullTorque)
{
}


TrapperLong::~TrapperLong()
{
}

//------------------------------------------------------------------------------

#if ( DIM == 2 )

/**
 Returns -len or +len
 */
real TrapperLong::calcArm(const PointInterpolated & pt, Vector const& pos, real len)
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
Vector TrapperLong::calcArm(const PointInterpolated & pt, Vector const& pos, real len)
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

Vector TrapperLong::posSide() const
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
Vector TrapperLong::force() const
{
    Vector d = cHand2->pos() - posSide();
    
    //correct for periodic space:
    if ( modulo )
        modulo->fold(d);
    
    return prop->stiffness * d;
}

void TrapperLong::setInteractions(Meca & meca) const
{
    PointInterpolated pt1 = cHand1->interpolation();
    PointInterpolated pt2 = cHand2->interpolation();
    
    if (trapped())
    {
        Hand * h_trap = trappedHand();
        if (h_trap->attached())
        {
            PointInterpolated * pi_same = 0, * pi_other = 0, ptt = h_trap->interpolation();
            if (h_trap->fiber()==cHand1->fiber())
            {
                pi_same = &pt1;
                pi_other = &pt2;
            }
            
            else if (h_trap->fiber()==cHand2->fiber())
            {
                pi_same = &pt2;
                pi_other = &pt1;
            }
            else
                std::cerr << "Trapping hand bound to third fiber in TrapperLong" << std::endl;
#if ( DIM == 2 )
            
            std::cerr << "TrapperLong not implemented in 2D" << std::endl;
            
#elif ( DIM == 3 )
            
            // Interaction with the trapper
            mArm = calcArm(*pi_other, ptt.pos(), prop->length);
            meca.interSideLinkS(*pi_other, ptt,  mArm, prop->length, prop->trap_stiffness);
            
            // Usual interaction, but being consistent with wich side of the couple acts as rigid
            // and which one extends
            mArm = calcArm(*pi_other, pi_same->pos(), prop->length);
            meca.interSideLinkS(*pi_other, *pi_same,  mArm, prop->length, prop->trap_stiffness);
            return;
#endif
        }
        
    }

    // The usual CoupleLong
#if ( DIM == 2 )
    
    std::cerr << "TrapperLong not implemented in 2D" << std::endl;
    
#elif ( DIM == 3 )
    
    mArm = calcArm(pt1, pt2.pos(), prop->length);
    meca.interSideLinkS(pt1, pt2, mArm, prop->length, prop->stiffness);
    //@todo Trapper::setInteractions() use interSideLink3D()
    
#endif
}

void TrapperLong::setInteractionsAF(Meca & meca) const
{
    if (trapped())
    {
        Hand * h_trap = trappedHand();
        if (h_trap->attached() && h_trap->fiber()==fiber1())
        {
#if (DIM!=1)
            PointInterpolated ptt = h_trap->interpolation();
            PointInterpolated pt1 = cHand1->interpolation();
            mArm = calcArm(pt1, ptt.pos(), prop->length);
            meca.interSideLinkS(pt1, ptt, mArm, prop->length, prop->stiffness);
#endif
        }
        
    }
}

void TrapperLong::setInteractionsFA(Meca & meca) const
{
    if (trapped())
    {
        Hand * h_trap = trappedHand();
        if (h_trap->attached() && h_trap->fiber()==fiber2())
        {
#if (DIM!=1)
            PointInterpolated ptt = h_trap->interpolation();
            PointInterpolated pt2 = cHand2->interpolation();
            mArm = calcArm(pt2, ptt.pos(), prop->length);
            meca.interSideLinkS(pt2, ptt, mArm, prop->length, prop->stiffness);
#endif
        }
        
    }
}






