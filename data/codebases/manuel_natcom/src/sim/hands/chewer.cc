// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#include "chewer.h"
#include "chewer_prop.h"
#include "glossary.h"
#include "exceptions.h"
#include "iowrapper.h"
#include "simul.h"

extern Random RNG;

//------------------------------------------------------------------------------

Chewer::Chewer(ChewerProp const* p, HandMonitor* h)
: Hand(p,h), prop(p)
{
    engaged = NO_END;
}


void Chewer::attach(FiberBinder const& fb)
{
    engaged = NO_END;
    Hand::attach(fb);
}


void Chewer::stepUnloaded()
{
    assert_true( attached() );
    
    // test for detachment
    if ( testDetachment() )
        return;
    
    if ( engaged != NO_END )
    {
        fbFiber->chew(prop->chewing_speed_dt, engaged);
        moveToEnd(engaged);
        return;
    }

    real e, a = fbAbs + prop->diffusion_dt * RNG.sreal();
    
    if ( a < fbFiber->abscissaC() )
    {
        e = fbFiber->abscissaM();
        
        if ( a <= e )
        {
            a = e;
            engaged = MINUS_END;
        }
    }
    else
    {
        e = fbFiber->abscissaP();
        
        if ( a >= e )
        {
            a = e;
            engaged = PLUS_END;
        }
    }
    
    // movement can lead to detachment, so we do it last:
    moveTo(a);
}


void Chewer::stepLoaded(Vector const& force)
{
    assert_true( attached() );
    
    if ( testKramersDetachment(force.norm()) )
        return;
    
    if ( engaged != NO_END )
    {
        fbFiber->chew(prop->chewing_speed_dt, engaged);
        moveToEnd(engaged);
        return;
    }
    
    // the load is the projection of the force on the local direction of Fiber
    real load = force * dirFiber();
    real e, a = fbAbs + prop->diffusion_dt * RNG.sreal() + prop->mobility_dt * load;
    
    if ( a < fbFiber->abscissaC() )
    {
        e = fbFiber->abscissaM();
        
        if ( a <= e )
        {
            a = e;
            engaged = MINUS_END;
        }
    }
    else
    {
        e = fbFiber->abscissaP();
        
        if ( a >= e )
        {
            a = e;
            engaged = PLUS_END;
        }
    }
    
    // movement can lead to detachment, so we do it last:
    moveTo(a);
}



