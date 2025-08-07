// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#include "duo.h"
#include "duo_prop.h"
#include "random.h"
#include "modulo.h"
#include "sim.h"

extern Random RNG;
extern Modulo const* modulo;

//------------------------------------------------------------------------------

Duo::Duo(DuoProp const* p, Vector const& w)
: Couple(p, w), mActive(0), prop(p)
{
}

Duo::~Duo()
{
    prop = 0;
}

//------------------------------------------------------------------------------

void Duo::activate()
{
    mActive = 1;
    gspTime = RNG.exponential();
}

void Duo::deactivate()
{
    mActive = 0;
}

//------------------------------------------------------------------------------

void Duo::stepFF(const FiberGrid& grid)
{
    // diffusion:
    cPos.addRand(prop->diffusion_dt);
    
    // confinement:
    if ( prop->confine == CONFINE_INSIDE )
    {
        if ( !prop->confine_space_ptr->inside(cPos) )
            prop->confine_space_ptr->bounce(cPos);
        if ( modulo )
            modulo->fold(cPos);
    }
    else if ( prop->confine == CONFINE_ON )
    {
        cPos = prop->confine_space_ptr->project(cPos);
    }    
    
    // check activity
    ///@todo better Duo::activation criteria
    if ( prop->activation_space_ptr->inside(cPos) )
        activate();

    
    // activity
    if ( mActive )
    {
        // spontaneous de-activation:
        gspTime -= prop->deactivation_rate_dt;
        if ( gspTime <= 0 )
        {
            deactivate();
            // test fraction of time when it is inactive:
            if ( RNG.test(-gspTime/prop->deactivation_rate_dt) )
                return;
        }
    
        // hands may bind:
        cHand1->stepUnattached(grid, cPos);
        if ( !prop->trans_activated )
            cHand2->stepUnattached(grid, cPos);
    }
}


/**
 test for spontaneous de-activation
 */
void Duo::deactivation()
{
    gspTime -= prop->deactivation_rate_dt;
    if ( gspTime <= 0 )
        deactivate();
}


/**
 Simulates:
 - attachment of cHand2
 - attached activity of cHand1
 .
 */
void Duo::stepAF(const FiberGrid& grid)
{
    if ( mActive && prop->vulnerable )
        deactivation();
    
    //we use cHand1->pos() first, because stepUnloaded() may detach cHand1
    cHand2->stepUnattached(grid, cHand1->pos());
    cHand1->stepUnloaded();
}


/**
 Simulates:
 - attachment of cHand1
 - attached activity of cHand2
 .
 */
void Duo::stepFA(const FiberGrid& grid)
{
    if ( mActive && prop->vulnerable )
        deactivation();
    
    //we use cHand2->pos() first, because stepUnloaded() may detach cHand2
    cHand1->stepUnattached(grid, cHand2->pos());
    cHand2->stepUnloaded();
}


/**
 Simulates:
 - attached activity of cHand1
 - attached activity of cHand2
 .
 */
void Duo::stepAA()
{
    if ( mActive && prop->vulnerable )
        deactivation();

    Vector f = force();
    cHand1->stepLoaded( f);
    cHand2->stepLoaded(-f);
}


//------------------------------------------------------------------------------

void Duo::write(Outputter& out) const
{
    out.writeUInt8(mActive);
    Couple::write(out);
}


void Duo::read(Inputter & in, Simul& sim, Tag tag)
{
#ifdef BACKWARD_COMPATIBILITY
    if ( in.formatID() > 36 )
#endif
    mActive = in.readUInt8();
    Couple::read(in, sim, tag);
}


