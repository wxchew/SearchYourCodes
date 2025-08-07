// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.

#ifndef DIGIT_H
#define DIGIT_H

#include "hand.h"

// Modest improvement of speed when length does not change.
//#define DIGITS_DEFINED_LENGTH
//#define DIGITS_GILLESPIE

class DigitProp;


/// A Hand that bind to discrete sites along a Fiber
/**
 The Digit is a Hand that can only bind at discrete positions along a Fiber.
 Adjacent binding sites are separated by @ref DigitPar "digit:step_size".
 
 If @ref DigitPar `digit:use_lattice = 1`, the Digit will use the Lattice
 associated to the fiber to limit the occupancy of a lattice site to 1:
 - the digit will bind only at an empty lattice site,
 - upon binding, the digit will occupy the corresponding site entirely
 .
 Note that with the current implementation, if ( `digit:use_lattice==1` ),
 then `digit:step_size` and `fiber:lattice_size` must be equal.
 
 See Examples and the @ref DigitPar.
 @ingroup HandGroup
 
 As defined in Hand, detachment increases exponentially with force.
 
 @todo handle cases where digit::step_size = INT * fiber:lattice_size
 */
class Digit : public Hand
{
    /// disabled default constructor
    Digit();
    
protected:
    
    /// index in the Lattice of attachement point
    int            mSite;
    
    /// copy of the Lattice of the fiber to which this is attached
    FiberLattice * mLattice;
    
    /// relocate to given site and update lattice
    void           resite(int npos);
    
    /// return the Lattice of the Fiber, if use_lattice is true
    FiberLattice * getLattice(const Fiber *) const;
    
public:
    
    /// Property
    DigitProp const* prop;
    
    /// constructor
    Digit(DigitProp const* p, HandMonitor* h);
    
    /// destructor
    ~Digit(){};
    
    
    /// check if attachement is possible according to properties
    int   attachmentAllowed(FiberBinder& site) const;
    
#ifdef MULTI_LATTICE
#ifdef MANU_LATTICE
    /// In case redistribution is allowed
    bool   attachmentSecondTry(FiberBinder& site);
#endif
#endif
    /// attach and update variables
    void   attach(FiberBinder const& site);
    
    /// detach
    void   detach();
    
    
    /// change the location
    void   moveTo(real a);
    
    /// move to the Minus End
    void   moveToEndM();
    
    /// move to the Plus End
    void   moveToEndP();
    
    /// change the fiber, and also the lattice if necessary
    void   relocate(Fiber* f);
    
    /// change the fiber, and also the lattice if necessary
    void   relocate(Fiber* f, real a);
    
    
    /// relocate without checking intermediate sites
    int    jumpTo(int npos);
    
    /// relocate without checking intermediate sites
    int    jumpToEnd(FiberEnd end);
    
    
    /// attempt one step towards the PLUS_END
    int    stepP();
    
    /// attempt one step towards the MINUS_END
    int    stepM();
    
    
    /// attempt one step of size `s` towards the PLUS_END
    int    jumpP(int s);
    
    /// attempt one step of size `s` towards the MINUS_END
    int    jumpM(int s);
    
    
    /// attempt `n` steps towards the PLUS_END, checking all intermediate sites
    int    crawlP(int n);
    
    /// attempt `n` steps towards the MINUS_END, checking all intermediate sites
    int    crawlM(int n);
    
    
    /// simulate when `this` is attached but not under load
    void   stepUnloaded();
    
    /// simulate when `this` is attached and under load
    void   stepLoaded(Vector const & force);
    
    
    /// this is called when the attachment point is beyond the PLUS_END
    void   handleDisassemblyM();
    
    /// this is called when the attachment point is below the MINUS_END
    void   handleDisassemblyP();
    
    /// read
    void   read(Inputter&, Simul&);
    
    /// write
    void   write(Outputter&) const;
    
#ifdef MULTI_LATTICE
    /// this indicates to which lattice it belongs to
    unsigned int lat_id;
    /// this sets the multilattice value, takes value from 1 to 4, and adds the value of prop->lat_val if necessary.
    void    set_lat_id(unsigned int const val);
    /// this gets the multilattice value used for the fibers (1 to 4), substracting the lat_val if necessary
    unsigned int    get_fiberlat_id() const;
    /// special step to set the random number
    void    stepUnattached(const FiberGrid&, Vector const &);
    
    ///
    void    random_multi_lattice(){set_lat_id(RNG.pint(4)+1);};
    
#ifdef MANU_LATTICE
    int lattice_val_add() const ;
#endif
    
#endif
    
#ifdef MANU_LATTICE
    bool find_lattice_neighbour(unsigned int const v) const;
#endif
    /// Check whether a position is free, not valid for first or last position in the lattice
    int checkPos(int npos);
    /// Check whether position towards the minus end is free
    int checkM();
    /// Check whether position towards the plus end is free
    int checkP();
    
#ifdef DIGITS_DEFINED_LENGTH
    /// Maximum abscissa defined when attached
    int max_abscissa;
#endif
    
#ifdef DIGITS_GILLESPIE
    /// The firing time
    real firing_t;
    
    /// Set that firing time
    virtual real set_fire(){return INFINITY;};
    
    virtual real set_fire_force(real f){return INFINITY;};
#endif
    int lattice_site() const {return mSite;};
#ifdef TRAP_SINGLES
    void stepUnattachedTrappedAA();
#endif
    
};

#endif

