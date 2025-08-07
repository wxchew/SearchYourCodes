// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#ifndef CROSSLINK_H
#define CROSSLINK_H

#include "couple.h"
class CrosslinkProp;

/// A specialized kind of Couple
/**
 The Crosslink is a simpler kind of Couple, which does not support:
 - specificity
 - trans_activated
 .
 It has a zero resting length, and uses Meca:interLink()
 
 CrosslinkLong has a non-zero resting length, and is selected automatically 
 @ingroup CoupleGroup
 */
class Crosslink : public Couple
{
public:
    
    /// property
    CrosslinkProp const* prop;
    
    /// create following the specifications in the CoupleProp
    Crosslink(CrosslinkProp const*, Vector const & w = Vector(0,0,0));

    /// destructor
    virtual      ~Crosslink();

    /// specialization of HandMonitor
    bool          allowAttachment(const FiberBinder &);
    
    /// simulation step for a free Couple: diffusion
    virtual void  stepFF(const FiberGrid&);
    
    /// add interactions to the Meca
    void          setInteractions(Meca &) const;

};


#endif

