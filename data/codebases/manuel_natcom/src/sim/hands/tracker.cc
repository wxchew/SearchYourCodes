// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#include "tracker.h"
#include "tracker_prop.h"
#include "glossary.h"
#include "exceptions.h"
#include "iowrapper.h"
#include "simul.h"
extern Random RNG;


Tracker::Tracker(TrackerProp const* p, HandMonitor* h)
: Hand(p, h), prop(p)
{
}


int Tracker::attachmentAllowed(FiberBinder & fb) const
{
    if ( !Hand::attachmentAllowed(fb) )
        return false;

    // check if fiber end in growing:
    if ( prop->bind_only_growing_end && !fb.fiber()->isGrowing(fb.nearestEnd()) )
        return false;
    
    return true;
}

//------------------------------------------------------------------------------
#pragma mark -


void Tracker::stepUnloaded()
{
    assert_true( attached() );
    
    // detachment
    if ( testDetachment() )
        return;

    
    switch ( prop->track_end )
    {
        case NO_END:
            break;
            
        case PLUS_END:
            moveToEndP();
            break;
            
        case MINUS_END:
            moveToEndM();
            break;
            
        case BOTH_ENDS:
            moveToEnd(nearestEnd());
            break;
            
        default:
            throw InvalidParameter("invalid value of tracker:track_end");
    }
}



void Tracker::stepLoaded(Vector const& force)
{
    assert_true( attached() );
    
    if ( testKramersDetachment(force.norm()) )
        return;

    switch ( prop->track_end )
    {
        case NO_END:
            break;
            
        case PLUS_END:
            moveToEndP();
            break;
            
        case MINUS_END:
            moveToEndM();
            break;
            
        case BOTH_ENDS:
            moveToEnd(nearestEnd());
            break;
            
        default:
            throw InvalidParameter("invalid value of tracker:track_end");
    }
}


