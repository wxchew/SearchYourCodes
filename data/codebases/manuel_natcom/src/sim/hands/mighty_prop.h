// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#ifndef MIGHTY_PROP_H
#define MIGHTY_PROP_H

#include "hand_prop.h"

/// additional Property for Mighty
/**
 @ingroup Properties
 */
class MightyProp : public HandProp
{
    friend class Mighty;

public:
    
    /**
     @defgroup MightyPar Parameters of Mighty
     @ingroup Parameters
     Inherits @ref HandPar.
     @{
     */
    
    
    /// force at which speed becomes zero
    real     stall_force;
    
    /// speed if force=0
    real     unloaded_speed;
    
    /// if true, speed is limited to [0, 2*max_speed]
    /**
     With ( limit_speed = 1 ), a plus-end directed motor will not move towards the minus-end,
     and also will not exceed 2x its max speed.
     For a minus-end directed motor, the permitted range is [2*unloaded_speed, 0],
     thus also excluding backward steps.
     */
    bool     limit_speed;
    
    /// density of detachment associated with active motion
    real     unbinding_density;
    
    /// @}
    
private:
    
    /// limits for a displacement in one time_step apply if ( limit_speed = true )
    real    min_dabs, max_dabs;
    
    /// variables derived from `unloaded_speed`
    real    set_speed_dt, abs_speed_dt, var_speed_dt;

public:
        
    /// constructor
    MightyProp(const std::string& n) : HandProp(n)  { clear(); }
   
    /// destructor
    ~MightyProp() { }
    
    /// return a Hand with this property
    virtual Hand * newHand(HandMonitor* h) const;

    /// set default values
    void clear();
        
    /// set from a Glossary
    void read(Glossary&);
    
    /// compute values derived from the parameters
    void complete(Simul const*);
    
    /// perform additional tests for the validity of parameters, given the elasticity
    void checkStiffness(real stiff, real len, real mul, real kT) const;
    
    /// return a carbon copy of object
    Property* clone() const { return new MightyProp(*this); }

    /// write all values
    void write_values(std::ostream &) const;
    
};

#endif

