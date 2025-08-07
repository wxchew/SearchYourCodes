// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#include "crosslink.h"
#include "crosslink_prop.h"
#include "exceptions.h"
#include "random.h"
#include "modulo.h"
#include "meca.h"

extern Random RNG;
extern Modulo const* modulo;

//------------------------------------------------------------------------------
Crosslink::Crosslink(CrosslinkProp const* p, Vector const& w)
: Couple(p, w), prop(p)
{
}


Crosslink::~Crosslink()
{
    prop = 0;
}


//------------------------------------------------------------------------------
#pragma mark -


/**
 @return:
 - True if attachment is possible
 - False if attachment is forbiden
 .
 
 If ( couple:stiff == true ), the two Hands of the Couple will refuse to be attached
 to the same segment, or to two neighboring segments on the same fiber.
 
 We cannot calculate the force of such 'degenerate' links, and they are undesired in
 most cases.
 
 */

bool Crosslink::allowAttachment(const FiberBinder & fb)
{
    Hand const* that = attachedHand();
    
    if ( that == 0 )
        return true;
    
    if ( prop->stiff )
    {
        if ( that->fiber() == fb.fiber()
            && fabs(fb.abscissa()-that->abscissa()) < 2*fb.fiber()->segmentation() )
            return false;
    }
    
    //attachment is allowed by default:
    return true;
}


/**
 Simulates:
 - diffusive motion
 - attachment
 .
 */
void Crosslink::stepFF(const FiberGrid& grid)
{
    // diffusive motion:
    cPos.addRand(prop->diffusion_dt);
    
    // confinement:
    if ( !prop->confine_space_ptr->inside(cPos) )
        prop->confine_space_ptr->bounce(cPos);
    if ( modulo )
        modulo->fold(cPos);
    
    // activity:
    cHand1->stepUnattached(grid, cPos);
    cHand2->stepUnattached(grid, cPos);
}



void Crosslink::setInteractions(Meca & meca) const
{
    assert_true( cHand1->attached() && cHand2->attached() );
    
    meca.interLink(cHand1->interpolation(), cHand2->interpolation(), prop->stiffness);
}

