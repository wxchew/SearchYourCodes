// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#include "digit.h"
#include "walker.h"
#include "walker_prop.h"
#include "glossary.h"
#include "lattice.h"
#include "simul.h"
extern Random RNG;


//------------------------------------------------------------------------------

Walker::Walker(WalkerProp const* p, HandMonitor* h)
: Digit(p,h), nextStep(0), prop(p)
{
    //These should be properties, also because otherwise you cannot change them with a "change hand" call within the config file
//    plusEndDirected = ( prop->unloaded_speed > 0 );
//    step_val = plusEndDirected ? 1:-1;
}


//------------------------------------------------------------------------------

void Walker::attach(FiberBinder const& fb)
{
    Digit::attach(fb);
    nextStep = RNG.exponential();
}


/**
 Currently, the Walker only makes forward steps, but backward steps exist as well.
 \todo simulate occurence of backward steps
 */
void Walker::stepUnloaded()
{
    assert_true( attached() );
    
    nextStep   -= prop->stepping_rate_dt;
    while ( nextStep <= 0 )
    {
        assert_true( attached() );
        // test detachment due to stepping
        if ( RNG.test(prop->unbinding_chance) )
        {
            detach();
            return;
        }
        
        if ( stepForward() == 2 )
        {
            // we have encountered the tip of the fiber
            if ( RNG.test_not(prop->dangling_chance) )
            {
                detach();
                return;
            }
        }
        nextStep += RNG.exponential();
    }
    
    testDetachment();
}



/**
 Currently, antagonistic force only reduces the rate of forward stepping.
 However, force is also known to increase the rate of backward steps.
 \todo simulate occurence of backward steps in Walker
 */
void Walker::stepLoaded(Vector const& force)
{
    assert_true( attached() );
    
    // calculate displacement, dependent on the load along the desired direction of displacement
    real rate_step = prop->stepping_rate_dt + ( force * dirFiber() ) * prop->var_rate_dt;

    nextStep -= rate_step;
    
    while ( nextStep <= 0)
    {
        // test detachment due to stepping
        if ( prop->unbinding_chance>0 && RNG.test(prop->unbinding_chance) )
        {
            detach();
            return;
        }
        
        if ( stepForward() == 2 )
        {
            // we have encountered the tip of the fiber
            if ( RNG.test_not(prop->dangling_chance) )
            {
                detach();
                return;
            }
        }
        nextStep += RNG.exponential();
    }
    
    if ( prop->unbinding_force_inv > 0 )
        testKramersDetachment(force.norm());
    else
        testDetachment();
}

/// attempts a step towards PLUS_END if ( walker:max_speed > 0 )
int  Walker::stepForward()  { return ( prop->plusEndDirected ? Digit::stepP() : Digit::stepM() ); }

/// attempts a step towards MINUS_END if ( walker:max_speed > 0 )
int  Walker::stepBackward() { return ( prop->plusEndDirected ? Digit::stepM() : Digit::stepP() ); }

/// Checks whether the next place is free
int  Walker::checkNext() {return ( prop->plusEndDirected ? Digit::checkP() : Digit::checkM() );}

