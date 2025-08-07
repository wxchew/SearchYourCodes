// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#ifndef REGULATOR_H
#define REGULATOR_H

#include "hand.h"

class RegulatorProp;

/// A Hand that may induce a catastrophe.
/**
 The Regulator is a Hand, and as such can bind and unbind from fibers.
 
 This class is not implemented.
 
 See Examples and the @ref RegulatorPar.
 @ingroup HandGroup
 */
class Regulator : public Hand
{
private:
    
    /// disabled default constructor
    Regulator();
    
public:
    
    /// Property
    RegulatorProp const* prop;
    
    /// constructor
    Regulator(RegulatorProp const* p, HandMonitor* h);
    
    /// destructor
    ~Regulator() {}
    
    /// attach the hand at the position described by site
    void   attach(FiberBinder const& site);
    
    /// simulate when `this` is attached but not under load
    void   stepUnloaded();
    
    /// simulate when `this` is attached and under load
    void   stepLoaded(Vector const & force);
    
};

#endif

