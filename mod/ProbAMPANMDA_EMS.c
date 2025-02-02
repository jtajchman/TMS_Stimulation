/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__ProbAMPANMDA_EMS
#define _nrn_initial _nrn_initial__ProbAMPANMDA_EMS
#define nrn_cur _nrn_cur__ProbAMPANMDA_EMS
#define _nrn_current _nrn_current__ProbAMPANMDA_EMS
#define nrn_jacob _nrn_jacob__ProbAMPANMDA_EMS
#define nrn_state _nrn_state__ProbAMPANMDA_EMS
#define _net_receive _net_receive__ProbAMPANMDA_EMS 
#define setRNG setRNG__ProbAMPANMDA_EMS 
#define state state__ProbAMPANMDA_EMS 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define tau_d_AMPA _p[0]
#define tau_d_AMPA_columnindex 0
#define Use _p[1]
#define Use_columnindex 1
#define Dep _p[2]
#define Dep_columnindex 2
#define Fac _p[3]
#define Fac_columnindex 3
#define mg _p[4]
#define mg_columnindex 4
#define u0 _p[5]
#define u0_columnindex 5
#define Nrrp _p[6]
#define Nrrp_columnindex 6
#define synapseID _p[7]
#define synapseID_columnindex 7
#define verboseLevel _p[8]
#define verboseLevel_columnindex 8
#define selected_for_report _p[9]
#define selected_for_report_columnindex 9
#define NMDA_ratio _p[10]
#define NMDA_ratio_columnindex 10
#define i _p[11]
#define i_columnindex 11
#define i_AMPA _p[12]
#define i_AMPA_columnindex 12
#define i_NMDA _p[13]
#define i_NMDA_columnindex 13
#define g_AMPA _p[14]
#define g_AMPA_columnindex 14
#define g_NMDA _p[15]
#define g_NMDA_columnindex 15
#define g _p[16]
#define g_columnindex 16
#define A_AMPA_step _p[17]
#define A_AMPA_step_columnindex 17
#define B_AMPA_step _p[18]
#define B_AMPA_step_columnindex 18
#define A_NMDA_step _p[19]
#define A_NMDA_step_columnindex 19
#define B_NMDA_step _p[20]
#define B_NMDA_step_columnindex 20
#define unoccupied _p[21]
#define unoccupied_columnindex 21
#define occupied _p[22]
#define occupied_columnindex 22
#define tsyn _p[23]
#define tsyn_columnindex 23
#define u _p[24]
#define u_columnindex 24
#define A_AMPA _p[25]
#define A_AMPA_columnindex 25
#define B_AMPA _p[26]
#define B_AMPA_columnindex 26
#define A_NMDA _p[27]
#define A_NMDA_columnindex 27
#define B_NMDA _p[28]
#define B_NMDA_columnindex 28
#define factor_AMPA _p[29]
#define factor_AMPA_columnindex 29
#define factor_NMDA _p[30]
#define factor_NMDA_columnindex 30
#define mggate _p[31]
#define mggate_columnindex 31
#define usingR123 _p[32]
#define usingR123_columnindex 32
#define DA_AMPA _p[33]
#define DA_AMPA_columnindex 33
#define DB_AMPA _p[34]
#define DB_AMPA_columnindex 34
#define DA_NMDA _p[35]
#define DA_NMDA_columnindex 35
#define DB_NMDA _p[36]
#define DB_NMDA_columnindex 36
#define v _p[37]
#define v_columnindex 37
#define _g _p[38]
#define _g_columnindex 38
#define _tsav _p[39]
#define _tsav_columnindex 39
#define _nd_area  *_ppvar[0]._pval
#define rng	*_ppvar[2]._pval
#define _p_rng	_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  2;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_bbsavestate(void*);
 static double _hoc_setRNG(void*);
 static double _hoc_state(void*);
 static double _hoc_toggleVerbose(void*);
 static double _hoc_urand(void*);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "bbsavestate", _hoc_bbsavestate,
 "setRNG", _hoc_setRNG,
 "state", _hoc_state,
 "toggleVerbose", _hoc_toggleVerbose,
 "urand", _hoc_urand,
 0, 0
};
#define bbsavestate bbsavestate_ProbAMPANMDA_EMS
#define toggleVerbose toggleVerbose_ProbAMPANMDA_EMS
#define urand urand_ProbAMPANMDA_EMS
 extern double bbsavestate( _threadargsproto_ );
 extern double toggleVerbose( _threadargsproto_ );
 extern double urand( _threadargsproto_ );
 /* declare global and static user variables */
#define e e_ProbAMPANMDA_EMS
 double e = 0;
#define gmax gmax_ProbAMPANMDA_EMS
 double gmax = 0.001;
#define tau_d_NMDA tau_d_NMDA_ProbAMPANMDA_EMS
 double tau_d_NMDA = 43;
#define tau_r_NMDA tau_r_NMDA_ProbAMPANMDA_EMS
 double tau_r_NMDA = 0.29;
#define tau_r_AMPA tau_r_AMPA_ProbAMPANMDA_EMS
 double tau_r_AMPA = 0.2;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "tau_r_AMPA_ProbAMPANMDA_EMS", "ms",
 "tau_r_NMDA_ProbAMPANMDA_EMS", "ms",
 "tau_d_NMDA_ProbAMPANMDA_EMS", "ms",
 "e_ProbAMPANMDA_EMS", "mV",
 "gmax_ProbAMPANMDA_EMS", "uS",
 "tau_d_AMPA", "ms",
 "Use", "1",
 "Dep", "ms",
 "Fac", "ms",
 "mg", "mM",
 "Nrrp", "1",
 "NMDA_ratio", "1",
 "i", "nA",
 "i_AMPA", "nA",
 "i_NMDA", "nA",
 "g_AMPA", "uS",
 "g_NMDA", "uS",
 "g", "uS",
 "unoccupied", "1",
 "occupied", "1",
 "tsyn", "ms",
 "u", "1",
 0,0
};
 static double A_NMDA0 = 0;
 static double A_AMPA0 = 0;
 static double B_NMDA0 = 0;
 static double B_AMPA0 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "tau_r_AMPA_ProbAMPANMDA_EMS", &tau_r_AMPA_ProbAMPANMDA_EMS,
 "tau_r_NMDA_ProbAMPANMDA_EMS", &tau_r_NMDA_ProbAMPANMDA_EMS,
 "tau_d_NMDA_ProbAMPANMDA_EMS", &tau_d_NMDA_ProbAMPANMDA_EMS,
 "e_ProbAMPANMDA_EMS", &e_ProbAMPANMDA_EMS,
 "gmax_ProbAMPANMDA_EMS", &gmax_ProbAMPANMDA_EMS,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"ProbAMPANMDA_EMS",
 "tau_d_AMPA",
 "Use",
 "Dep",
 "Fac",
 "mg",
 "u0",
 "Nrrp",
 "synapseID",
 "verboseLevel",
 "selected_for_report",
 "NMDA_ratio",
 0,
 "i",
 "i_AMPA",
 "i_NMDA",
 "g_AMPA",
 "g_NMDA",
 "g",
 "A_AMPA_step",
 "B_AMPA_step",
 "A_NMDA_step",
 "B_NMDA_step",
 "unoccupied",
 "occupied",
 "tsyn",
 "u",
 0,
 "A_AMPA",
 "B_AMPA",
 "A_NMDA",
 "B_NMDA",
 0,
 "rng",
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 40, _prop);
 	/*initialize range parameters*/
 	tau_d_AMPA = 1.7;
 	Use = 1;
 	Dep = 100;
 	Fac = 10;
 	mg = 1;
 	u0 = 0;
 	Nrrp = 1;
 	synapseID = 0;
 	verboseLevel = 0;
 	selected_for_report = 0;
 	NMDA_ratio = 0.71;
  }
 	_prop->param = _p;
 	_prop->param_size = 40;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 static void _net_receive(Point_process*, double*, double);
 static void _net_init(Point_process*, double*, double);
 static void bbcore_write(double*, int*, int*, int*, _threadargsproto_);
 extern void hoc_reg_bbcore_write(int, void(*)(double*, int*, int*, int*, _threadargsproto_));
 static void bbcore_read(double*, int*, int*, int*, _threadargsproto_);
 extern void hoc_reg_bbcore_read(int, void(*)(double*, int*, int*, int*, _threadargsproto_));
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _ProbAMPANMDA_EMS_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
   hoc_reg_bbcore_write(_mechtype, bbcore_write);
   hoc_reg_bbcore_read(_mechtype, bbcore_read);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 40, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "bbcorepointer");
 	hoc_register_cvode(_mechtype, _ode_count, 0, 0, 0);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_init[_mechtype] = _net_init;
 pnt_receive_size[_mechtype] = 4;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 ProbAMPANMDA_EMS ProbAMPANMDA_EMS.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "Probabilistic AMPA and NMDA receptor with presynaptic short-term plasticity";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int setRNG(_threadargsproto_);
static int state(_threadargsproto_);
 
/*VERBATIM*/

#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include "nrnran123.h"

double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);

 
static int  state ( _threadargsproto_ ) {
   A_AMPA = A_AMPA * A_AMPA_step ;
   B_AMPA = B_AMPA * B_AMPA_step ;
   A_NMDA = A_NMDA * A_NMDA_step ;
   B_NMDA = B_NMDA * B_NMDA_step ;
    return 0; }
 
static double _hoc_state(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 state ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t; {
   double _lresult , _lves , _loccu ;
 _args[1] = _args[0] ;
   _args[2] = _args[0] * NMDA_ratio ;
   if ( _args[0] <= 0.0  || t < 0.0 ) {
     
/*VERBATIM*/
        return;
 }
   if ( Fac > 0.0 ) {
     u = u * exp ( - ( t - tsyn ) / Fac ) ;
     }
   else {
     u = Use ;
     }
   if ( Fac > 0.0 ) {
     u = u + Use * ( 1.0 - u ) ;
     }
   {int  _lcounter ;for ( _lcounter = 0 ; _lcounter <= ( ((int) unoccupied ) - 1 ) ; _lcounter ++ ) {
     _args[3] = exp ( - ( t - tsyn ) / Dep ) ;
     _lresult = urand ( _threadargs_ ) ;
     if ( _lresult > _args[3] ) {
       occupied = occupied + 1.0 ;
       if ( verboseLevel > 0.0 ) {
          printf ( "Recovered! %f at time %g: Psurv = %g, urand=%g\n" , synapseID , t , _args[3] , _lresult ) ;
          }
       }
     } }
   _lves = 0.0 ;
   _loccu = occupied - 1.0 ;
   {int  _lcounter ;for ( _lcounter = 0 ; _lcounter <= ((int) _loccu ) ; _lcounter ++ ) {
     _lresult = urand ( _threadargs_ ) ;
     if ( _lresult < u ) {
       occupied = occupied - 1.0 ;
       _lves = _lves + 1.0 ;
       }
     } }
   unoccupied = Nrrp - occupied ;
   tsyn = t ;
   if ( _lves > 0.0 ) {
     A_AMPA = A_AMPA + _lves / Nrrp * _args[1] * factor_AMPA ;
     B_AMPA = B_AMPA + _lves / Nrrp * _args[1] * factor_AMPA ;
     A_NMDA = A_NMDA + _lves / Nrrp * _args[2] * factor_NMDA ;
     B_NMDA = B_NMDA + _lves / Nrrp * _args[2] * factor_NMDA ;
     if ( verboseLevel > 0.0 ) {
        printf ( "Release! %f at time %g: vals %g %g %g %g\n" , synapseID , t , A_AMPA , _args[1] , factor_AMPA , _args[0] ) ;
        }
     }
   else {
     if ( verboseLevel > 0.0 ) {
        printf ( " || SYN_ID: %f, release failure || " , synapseID ) ;
        }
     }
   } }
 
static void _net_init(Point_process* _pnt, double* _args, double _lflag) {
       double* _p = _pnt->_prop->param;
    Datum* _ppvar = _pnt->_prop->dparam;
    Datum* _thread = (Datum*)0;
    NrnThread* _nt = (NrnThread*)_pnt->_vnt;
 }
 
static int  setRNG ( _threadargsproto_ ) {
   
/*VERBATIM*/
    #ifndef CORENEURON_BUILD
    // For compatibility, allow for either MCellRan4 or Random123
    // Distinguish by the arg types
    // Object => MCellRan4, seeds (double) => Random123
    usingR123 = 0;
    if( ifarg(1) && hoc_is_double_arg(1) ) {
        nrnran123_State** pv = (nrnran123_State**)(&_p_rng);
        uint32_t a2 = 0;
        uint32_t a3 = 0;

        if (*pv) {
            nrnran123_deletestream(*pv);
            *pv = (nrnran123_State*)0;
        }
        if (ifarg(2)) {
            a2 = (uint32_t)*getarg(2);
        }
        if (ifarg(3)) {
            a3 = (uint32_t)*getarg(3);
        }
        *pv = nrnran123_newstream3((uint32_t)*getarg(1), a2, a3);
        usingR123 = 1;
    } else if( ifarg(1) ) {   // not a double, so assume hoc object type
        void** pv = (void**)(&_p_rng);
        *pv = nrn_random_arg(1);
    } else {  // no arg, so clear pointer
        void** pv = (void**)(&_p_rng);
        *pv = (void*)0;
    }
    #endif
  return 0; }
 
static double _hoc_setRNG(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 setRNG ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
double urand ( _threadargsproto_ ) {
   double _lurand;
 
/*VERBATIM*/
    double value = 0.0;
    if ( usingR123 ) {
        value = nrnran123_dblpick((nrnran123_State*)_p_rng);
    } else if (_p_rng) {
        #ifndef CORENEURON_BUILD
        value = nrn_random_pick(_p_rng);
        #endif
    } else {
        // Note: prior versions used scop_random(1), but since we never use this model without configuring the rng.  Maybe should throw error?
        value = 0.0;
    }
    _lurand = value;
 
return _lurand;
 }
 
static double _hoc_urand(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  urand ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
double bbsavestate ( _threadargsproto_ ) {
   double _lbbsavestate;
 _lbbsavestate = 0.0 ;
   
/*VERBATIM*/
#ifndef CORENEURON_BUILD
        /* first arg is direction (0 save, 1 restore), second is array*/
        /* if first arg is -1, fill xdir with the size of the array */
        double *xdir, *xval, *hoc_pgetarg();
        long nrn_get_random_sequence(void* r);
        void nrn_set_random_sequence(void* r, int val);
        xdir = hoc_pgetarg(1);
        xval = hoc_pgetarg(2);
        if (_p_rng) {
            // tell how many items need saving
            if (*xdir == -1) {  // count items
                if( usingR123 ) {
                    *xdir = 2.0;
                } else {
                    *xdir = 1.0;
                }
                return 0.0;
            } else if(*xdir ==0 ) {  // save
                if( usingR123 ) {
                    uint32_t seq;
                    char which;
                    nrnran123_getseq( (nrnran123_State*)_p_rng, &seq, &which );
                    xval[0] = (double) seq;
                    xval[1] = (double) which;
                } else {
                    xval[0] = (double)nrn_get_random_sequence(_p_rng);
                }
            } else {  // restore
                if( usingR123 ) {
                    nrnran123_setseq( (nrnran123_State*)_p_rng, (uint32_t)xval[0], (char)xval[1] );
                } else {
                    nrn_set_random_sequence(_p_rng, (long)(xval[0]));
                }
            }
        }
#endif
 
return _lbbsavestate;
 }
 
static double _hoc_bbsavestate(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  bbsavestate ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
double toggleVerbose ( _threadargsproto_ ) {
   double _ltoggleVerbose;
 verboseLevel = 1.0 - verboseLevel ;
   
return _ltoggleVerbose;
 }
 
static double _hoc_toggleVerbose(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  toggleVerbose ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
/*VERBATIM*/
static void bbcore_write(double* x, int* d, int* xx, int* offset, _threadargsproto_) {
  if (d) {
    uint32_t* di = ((uint32_t*)d) + *offset;
    nrnran123_State** pv = (nrnran123_State**)(&_p_rng);
    nrnran123_getids3(*pv, di, di+1, di+2);

    char which;
    nrnran123_getseq(*pv, di+3, &which);
    di[4] = (int)which;
    //printf("ProbAMPANMDA_EMS bbcore_write %d %d %d\n", di[0], di[1], di[2]);
    }
  *offset += 5;
}

static void bbcore_read(double* x, int* d, int* xx, int* offset, _threadargsproto_) {
  assert(!_p_rng);
  uint32_t* di = ((uint32_t*)d) + *offset;
  if (di[0] != 0 || di[1] != 0 || di[2] != 0) {
      nrnran123_State** pv = (nrnran123_State**)(&_p_rng);
      *pv = nrnran123_newstream3(di[0], di[1], di[2]);
      unsigned char which = (unsigned char)di[4];
      nrnran123_setseq(*pv, di[3], which);
  }
  //printf("ProbAMPANMDA_EMS bbcore_read %d %d %d\n", di[0], di[1], di[2]);
  *offset += 5;
}
 
static int _ode_count(int _type){ hoc_execerror("ProbAMPANMDA_EMS", "cannot be used with CVODE"); return 0;}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  A_NMDA = A_NMDA0;
  A_AMPA = A_AMPA0;
  B_NMDA = B_NMDA0;
  B_AMPA = B_AMPA0;
 {
   double _ltp_AMPA , _ltp_NMDA ;
 tsyn = 0.0 ;
   u = u0 ;
   unoccupied = 0.0 ;
   occupied = Nrrp ;
   A_AMPA = 0.0 ;
   B_AMPA = 0.0 ;
   A_NMDA = 0.0 ;
   B_NMDA = 0.0 ;
   _ltp_AMPA = ( tau_r_AMPA * tau_d_AMPA ) / ( tau_d_AMPA - tau_r_AMPA ) * log ( tau_d_AMPA / tau_r_AMPA ) ;
   _ltp_NMDA = ( tau_r_NMDA * tau_d_NMDA ) / ( tau_d_NMDA - tau_r_NMDA ) * log ( tau_d_NMDA / tau_r_NMDA ) ;
   factor_AMPA = - exp ( - _ltp_AMPA / tau_r_AMPA ) + exp ( - _ltp_AMPA / tau_d_AMPA ) ;
   factor_AMPA = 1.0 / factor_AMPA ;
   factor_NMDA = - exp ( - _ltp_NMDA / tau_r_NMDA ) + exp ( - _ltp_NMDA / tau_d_NMDA ) ;
   factor_NMDA = 1.0 / factor_NMDA ;
   A_AMPA_step = exp ( dt * ( ( - 1.0 ) / tau_r_AMPA ) ) ;
   B_AMPA_step = exp ( dt * ( ( - 1.0 ) / tau_d_AMPA ) ) ;
   A_NMDA_step = exp ( dt * ( ( - 1.0 ) / tau_r_NMDA ) ) ;
   B_NMDA_step = exp ( dt * ( ( - 1.0 ) / tau_d_NMDA ) ) ;
   
/*VERBATIM*/
        if( usingR123 ) {
            nrnran123_setseq((nrnran123_State*)_p_rng, 0, 0);
        }
 }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _tsav = -1e20;
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   mggate = 1.0 / ( 1.0 + exp ( 0.062 * - ( v ) ) * ( mg / 3.57 ) ) ;
   g_AMPA = gmax * ( B_AMPA - A_AMPA ) ;
   g_NMDA = gmax * ( B_NMDA - A_NMDA ) * mggate ;
   g = g_AMPA + g_NMDA ;
   i_AMPA = g_AMPA * ( v - e ) ;
   i_NMDA = g_NMDA * ( v - e ) ;
   i = i_AMPA + i_NMDA ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
 {  { state(_p, _ppvar, _thread, _nt); }
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "ProbAMPANMDA_EMS.mod";
static const char* nmodl_file_text = 
  "COMMENT\n"
  "/**\n"
  " * @file ProbAMPANMDA_EMS.mod\n"
  " * @brief\n"
  " * @author king, muller, reimann, ramaswamy\n"
  " * @date 2011-08-17\n"
  " * @remark Copyright \n"
  "\n"
  " BBP/EPFL 2005-2011; All rights reserved. Do not distribute without further notice.\n"
  " */\n"
  "ENDCOMMENT\n"
  "\n"
  "TITLE Probabilistic AMPA and NMDA receptor with presynaptic short-term plasticity\n"
  "\n"
  "\n"
  "COMMENT\n"
  "AMPA and NMDA receptor conductance using a dual-exponential profile\n"
  "presynaptic short-term plasticity as in Fuhrmann et al. 2002\n"
  "\n"
  "_EMS (Eilif Michael Srikanth)\n"
  "Modification of ProbAMPANMDA: 2-State model by Eilif Muller, Michael Reimann, Srikanth Ramaswamy, Blue Brain Project, August 2011\n"
  "This new model was motivated by the following constraints:\n"
  "\n"
  "1) No consumption on failure.\n"
  "2) No release just after release until recovery.\n"
  "3) Same ensemble averaged trace as deterministic/canonical Tsodyks-Markram\n"
  "   using same parameters determined from experiment.\n"
  "4) Same quantal size as present production probabilistic model.\n"
  "\n"
  "To satisfy these constaints, the synapse is implemented as a\n"
  "uni-vesicular (generalization to multi-vesicular should be\n"
  "straight-forward) 2-state Markov process.  The states are\n"
  "{1=recovered, 0=unrecovered}.\n"
  "\n"
  "For a pre-synaptic spike or external spontaneous release trigger\n"
  "event, the synapse will only release if it is in the recovered state,\n"
  "and with probability u (which follows facilitation dynamics).  If it\n"
  "releases, it will transition to the unrecovered state.  Recovery is as\n"
  "a Poisson process with rate 1/Dep.\n"
  "\n"
  "This model satisys all of (1)-(4).\n"
  "\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "NEURON {\n"
  "    THREADSAFE\n"
  "    POINT_PROCESS ProbAMPANMDA_EMS\n"
  "\n"
  "    GLOBAL tau_r_AMPA\n"
  "    RANGE tau_d_AMPA, g_AMPA, i_AMPA\n"
  "\n"
  "    GLOBAL tau_r_NMDA, tau_d_NMDA\n"
  "    RANGE g_NMDA, i_NMDA\n"
  "\n"
  "    RANGE Use, u, Dep, Fac, u0, mg, tsyn\n"
  "    RANGE unoccupied, occupied, Nrrp\n"
  "\n"
  "    RANGE g, NMDA_ratio\n"
  "    RANGE A_AMPA_step, B_AMPA_step, A_NMDA_step, B_NMDA_step\n"
  "    GLOBAL e\n"
  "    NONSPECIFIC_CURRENT i\n"
  "    BBCOREPOINTER rng\n"
  "    RANGE synapseID, selected_for_report, verboseLevel\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "        tau_r_AMPA = 0.2   (ms)  : dual-exponential conductance profile\n"
  "        tau_d_AMPA = 1.7    (ms)  : IMPORTANT: tau_r < tau_d\n"
  "        tau_r_NMDA = 0.29   (ms) : dual-exponential conductance profile\n"
  "        tau_d_NMDA = 43     (ms) : IMPORTANT: tau_r < tau_d\n"
  "        Use = 1.0   (1)   : Utilization of synaptic efficacy (just initial values! Use, Dep and Fac are overwritten by BlueBuilder assigned values)\n"
  "        Dep = 100   (ms)  : relaxation time constant from depression\n"
  "        Fac = 10   (ms)  :  relaxation time constant from facilitation\n"
  "        e = 0     (mV)  : AMPA and NMDA reversal potential\n"
  "        mg = 1   (mM)  : initial concentration of mg2+\n"
  "        gmax = .001 (uS) : weight conversion factor (from nS to uS)\n"
  "        u0 = 0 :initial value of u, which is the running value of release probability\n"
  "        Nrrp = 1 (1)  : Number of total release sites for given contact\n"
  "        synapseID = 0\n"
  "        verboseLevel = 0\n"
  "        selected_for_report = 0\n"
  "        NMDA_ratio = 0.71 (1) : The ratio of NMDA to AMPA\n"
  "}\n"
  "\n"
  "COMMENT\n"
  "The Verbatim block is needed to generate random nos. from a uniform distribution between 0 and 1\n"
  "for comparison with Pr to decide whether to activate the synapse or not\n"
  "ENDCOMMENT\n"
  "\n"
  "VERBATIM\n"
  "\n"
  "#include<stdlib.h>\n"
  "#include<stdio.h>\n"
  "#include<math.h>\n"
  "#include \"nrnran123.h\"\n"
  "\n"
  "double nrn_random_pick(void* r);\n"
  "void* nrn_random_arg(int argpos);\n"
  "\n"
  "ENDVERBATIM\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "\n"
  "        v (mV)\n"
  "        i (nA)\n"
  "        i_AMPA (nA)\n"
  "        i_NMDA (nA)\n"
  "        g_AMPA (uS)\n"
  "        g_NMDA (uS)\n"
  "        g (uS)\n"
  "        factor_AMPA\n"
  "        factor_NMDA\n"
  "        A_AMPA_step\n"
  "        B_AMPA_step\n"
  "        A_NMDA_step\n"
  "        B_NMDA_step\n"
  "        rng\n"
  "        mggate\n"
  "        usingR123            : TEMPORARY until mcellran4 completely deprecated\n"
  "\n"
  "        : MVR\n"
  "        unoccupied (1) : no. of unoccupied sites following release event\n"
  "        occupied   (1) : no. of occupied sites following one epoch of recovery\n"
  "        tsyn (ms) : the time of the last spike\n"
  "        u (1) : running release probability\n"
  "}\n"
  "\n"
  "\n"
  "STATE {\n"
  "\n"
  "        A_AMPA       : AMPA state variable to construct the dual-exponential profile - decays with conductance tau_r_AMPA\n"
  "        B_AMPA       : AMPA state variable to construct the dual-exponential profile - decays with conductance tau_d_AMPA\n"
  "        A_NMDA       : NMDA state variable to construct the dual-exponential profile - decays with conductance tau_r_NMDA\n"
  "        B_NMDA       : NMDA state variable to construct the dual-exponential profile - decays with conductance tau_d_NMDA\n"
  "}\n"
  "\n"
  "\n"
  "INITIAL {\n"
  "        LOCAL tp_AMPA, tp_NMDA\n"
  "\n"
  "        tsyn = 0\n"
  "        u=u0\n"
  "\n"
  "        : MVR\n"
  "        unoccupied = 0\n"
  "        occupied = Nrrp\n"
  "\n"
  "        A_AMPA = 0\n"
  "        B_AMPA = 0\n"
  "\n"
  "        A_NMDA = 0\n"
  "        B_NMDA = 0\n"
  "\n"
  "        tp_AMPA = (tau_r_AMPA*tau_d_AMPA)/(tau_d_AMPA-tau_r_AMPA)*log(tau_d_AMPA/tau_r_AMPA) :time to peak of the conductance\n"
  "        tp_NMDA = (tau_r_NMDA*tau_d_NMDA)/(tau_d_NMDA-tau_r_NMDA)*log(tau_d_NMDA/tau_r_NMDA) :time to peak of the conductance\n"
  "\n"
  "        factor_AMPA = -exp(-tp_AMPA/tau_r_AMPA)+exp(-tp_AMPA/tau_d_AMPA) :AMPA Normalization factor - so that when t = tp_AMPA, gsyn = gpeak\n"
  "        factor_AMPA = 1/factor_AMPA\n"
  "\n"
  "        factor_NMDA = -exp(-tp_NMDA/tau_r_NMDA)+exp(-tp_NMDA/tau_d_NMDA) :NMDA Normalization factor - so that when t = tp_NMDA, gsyn = gpeak\n"
  "        factor_NMDA = 1/factor_NMDA\n"
  "\n"
  "        A_AMPA_step = exp(dt*(( - 1.0 ) / tau_r_AMPA))\n"
  "        B_AMPA_step = exp(dt*(( - 1.0 ) / tau_d_AMPA))\n"
  "        A_NMDA_step = exp(dt*(( - 1.0 ) / tau_r_NMDA))\n"
  "        B_NMDA_step = exp(dt*(( - 1.0 ) / tau_d_NMDA))\n"
  "\n"
  "        VERBATIM\n"
  "        if( usingR123 ) {\n"
  "            nrnran123_setseq((nrnran123_State*)_p_rng, 0, 0);\n"
  "        }\n"
  "        ENDVERBATIM\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "\n"
  "        SOLVE state\n"
  "        mggate = 1 / (1 + exp(0.062 (/mV) * -(v)) * (mg / 3.57 (mM))) :mggate kinetics - Jahr & Stevens 1990\n"
  "        g_AMPA = gmax*(B_AMPA-A_AMPA) :compute time varying conductance as the difference of state variables B_AMPA and A_AMPA\n"
  "        g_NMDA = gmax*(B_NMDA-A_NMDA) * mggate :compute time varying conductance as the difference of state variables B_NMDA and A_NMDA and mggate kinetics\n"
  "        g = g_AMPA + g_NMDA\n"
  "        i_AMPA = g_AMPA*(v-e) :compute the AMPA driving force based on the time varying conductance, membrane potential, and AMPA reversal\n"
  "        i_NMDA = g_NMDA*(v-e) :compute the NMDA driving force based on the time varying conductance, membrane potential, and NMDA reversal\n"
  "        i = i_AMPA + i_NMDA\n"
  "}\n"
  "\n"
  "PROCEDURE state() {\n"
  "        A_AMPA = A_AMPA*A_AMPA_step\n"
  "        B_AMPA = B_AMPA*B_AMPA_step\n"
  "        A_NMDA = A_NMDA*A_NMDA_step\n"
  "        B_NMDA = B_NMDA*B_NMDA_step\n"
  "}\n"
  "\n"
  "\n"
  "NET_RECEIVE (weight,weight_AMPA, weight_NMDA, Psurv) {\n"
  "    LOCAL result, ves, occu\n"
  "    weight_AMPA = weight\n"
  "    weight_NMDA = weight * NMDA_ratio\n"
  "    : Locals:\n"
  "    : Psurv - survival probability of unrecovered state\n"
  "\n"
  "    INITIAL {\n"
  "    }\n"
  "\n"
  "    : Do not perform any calculations if the synapse (netcon) is deactivated. This avoids drawing from\n"
  "    : random number stream. Also, disable in case of t < 0 (in case of ForwardSkip) which causes numerical\n"
  "    : instability if synapses are activated.\n"
  "    if(  weight <= 0 || t < 0 ) {\n"
  "    VERBATIM\n"
  "        return;\n"
  "    ENDVERBATIM\n"
  "    }\n"
  "\n"
  "    : calc u at event-\n"
  "    if (Fac > 0) {\n"
  "        u = u*exp(-(t - tsyn)/Fac) :update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.\n"
  "    } else {\n"
  "        u = Use\n"
  "    }\n"
  "    if(Fac > 0){\n"
  "        u = u + Use*(1-u) :update facilitation variable if Fac>0 Eq. 2 in Fuhrmann et al.\n"
  "    }\n"
  "\n"
  "    : recovery\n"
  "    FROM counter = 0 TO (unoccupied - 1) {\n"
  "        : Iterate over all unoccupied sites and compute how many recover\n"
  "        Psurv = exp(-(t-tsyn)/Dep)\n"
  "        result = urand()\n"
  "        if (result>Psurv) {\n"
  "            occupied = occupied + 1     : recover a previously unoccupied site\n"
  "            if( verboseLevel > 0 ) {\n"
  "                UNITSOFF\n"
  "                printf( \"Recovered! %f at time %g: Psurv = %g, urand=%g\\n\", synapseID, t, Psurv, result )\n"
  "                UNITSON\n"
  "            }\n"
  "        }\n"
  "    }\n"
  "\n"
  "    ves = 0                  : Initialize the number of released vesicles to 0\n"
  "    occu = occupied - 1  : Store the number of occupied sites in a local variable\n"
  "\n"
  "    FROM counter = 0 TO occu {\n"
  "        : iterate over all occupied sites and compute how many release\n"
  "        result = urand()\n"
  "        if (result<u) {\n"
  "            : release a single site!\n"
  "            occupied = occupied - 1  : decrease the number of occupied sites by 1\n"
  "            ves = ves + 1            : increase number of relesed vesicles by 1\n"
  "        }\n"
  "    }\n"
  "\n"
  "    : Update number of unoccupied sites\n"
  "    unoccupied = Nrrp - occupied\n"
  "\n"
  "    : Update tsyn\n"
  "    : tsyn knows about all spikes, not only those that released\n"
  "    : i.e. each spike can increase the u, regardless of recovered state.\n"
  "    :      and each spike trigger an evaluation of recovery\n"
  "    tsyn = t\n"
  "\n"
  "    if (ves > 0) { :no need to evaluate unless we have vesicle release\n"
  "        A_AMPA = A_AMPA + ves/Nrrp*weight_AMPA*factor_AMPA\n"
  "        B_AMPA = B_AMPA + ves/Nrrp*weight_AMPA*factor_AMPA\n"
  "        A_NMDA = A_NMDA + ves/Nrrp*weight_NMDA*factor_NMDA\n"
  "        B_NMDA = B_NMDA + ves/Nrrp*weight_NMDA*factor_NMDA\n"
  "\n"
  "        if ( verboseLevel > 0 ) {\n"
  "            UNITSOFF\n"
  "            printf( \"Release! %f at time %g: vals %g %g %g %g\\n\", synapseID, t, A_AMPA, weight_AMPA, factor_AMPA, weight )\n"
  "            UNITSON\n"
  "        }\n"
  "\n"
  "    } else {\n"
  "        : total release failure\n"
  "        if ( verboseLevel > 0 ) {\n"
  "            UNITSOFF\n"
  "            printf( \" || SYN_ID: %f, release failure || \", synapseID )\n"
  "            UNITSON\n"
  "        }\n"
  "    }\n"
  "}\n"
  "\n"
  "\n"
  "PROCEDURE setRNG() {\n"
  "VERBATIM\n"
  "    #ifndef CORENEURON_BUILD\n"
  "    // For compatibility, allow for either MCellRan4 or Random123\n"
  "    // Distinguish by the arg types\n"
  "    // Object => MCellRan4, seeds (double) => Random123\n"
  "    usingR123 = 0;\n"
  "    if( ifarg(1) && hoc_is_double_arg(1) ) {\n"
  "        nrnran123_State** pv = (nrnran123_State**)(&_p_rng);\n"
  "        uint32_t a2 = 0;\n"
  "        uint32_t a3 = 0;\n"
  "\n"
  "        if (*pv) {\n"
  "            nrnran123_deletestream(*pv);\n"
  "            *pv = (nrnran123_State*)0;\n"
  "        }\n"
  "        if (ifarg(2)) {\n"
  "            a2 = (uint32_t)*getarg(2);\n"
  "        }\n"
  "        if (ifarg(3)) {\n"
  "            a3 = (uint32_t)*getarg(3);\n"
  "        }\n"
  "        *pv = nrnran123_newstream3((uint32_t)*getarg(1), a2, a3);\n"
  "        usingR123 = 1;\n"
  "    } else if( ifarg(1) ) {   // not a double, so assume hoc object type\n"
  "        void** pv = (void**)(&_p_rng);\n"
  "        *pv = nrn_random_arg(1);\n"
  "    } else {  // no arg, so clear pointer\n"
  "        void** pv = (void**)(&_p_rng);\n"
  "        *pv = (void*)0;\n"
  "    }\n"
  "    #endif\n"
  "ENDVERBATIM\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION urand() {\n"
  "VERBATIM\n"
  "    double value = 0.0;\n"
  "    if ( usingR123 ) {\n"
  "        value = nrnran123_dblpick((nrnran123_State*)_p_rng);\n"
  "    } else if (_p_rng) {\n"
  "        #ifndef CORENEURON_BUILD\n"
  "        value = nrn_random_pick(_p_rng);\n"
  "        #endif\n"
  "    } else {\n"
  "        // Note: prior versions used scop_random(1), but since we never use this model without configuring the rng.  Maybe should throw error?\n"
  "        value = 0.0;\n"
  "    }\n"
  "    _lurand = value;\n"
  "ENDVERBATIM\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION bbsavestate() {\n"
  "        bbsavestate = 0\n"
  "VERBATIM\n"
  "#ifndef CORENEURON_BUILD\n"
  "        /* first arg is direction (0 save, 1 restore), second is array*/\n"
  "        /* if first arg is -1, fill xdir with the size of the array */\n"
  "        double *xdir, *xval, *hoc_pgetarg();\n"
  "        long nrn_get_random_sequence(void* r);\n"
  "        void nrn_set_random_sequence(void* r, int val);\n"
  "        xdir = hoc_pgetarg(1);\n"
  "        xval = hoc_pgetarg(2);\n"
  "        if (_p_rng) {\n"
  "            // tell how many items need saving\n"
  "            if (*xdir == -1) {  // count items\n"
  "                if( usingR123 ) {\n"
  "                    *xdir = 2.0;\n"
  "                } else {\n"
  "                    *xdir = 1.0;\n"
  "                }\n"
  "                return 0.0;\n"
  "            } else if(*xdir ==0 ) {  // save\n"
  "                if( usingR123 ) {\n"
  "                    uint32_t seq;\n"
  "                    char which;\n"
  "                    nrnran123_getseq( (nrnran123_State*)_p_rng, &seq, &which );\n"
  "                    xval[0] = (double) seq;\n"
  "                    xval[1] = (double) which;\n"
  "                } else {\n"
  "                    xval[0] = (double)nrn_get_random_sequence(_p_rng);\n"
  "                }\n"
  "            } else {  // restore\n"
  "                if( usingR123 ) {\n"
  "                    nrnran123_setseq( (nrnran123_State*)_p_rng, (uint32_t)xval[0], (char)xval[1] );\n"
  "                } else {\n"
  "                    nrn_set_random_sequence(_p_rng, (long)(xval[0]));\n"
  "                }\n"
  "            }\n"
  "        }\n"
  "#endif\n"
  "ENDVERBATIM\n"
  "}\n"
  "\n"
  "FUNCTION toggleVerbose() {\n"
  "    verboseLevel = 1-verboseLevel\n"
  "}\n"
  "\n"
  "\n"
  "VERBATIM\n"
  "static void bbcore_write(double* x, int* d, int* xx, int* offset, _threadargsproto_) {\n"
  "  if (d) {\n"
  "    uint32_t* di = ((uint32_t*)d) + *offset;\n"
  "    nrnran123_State** pv = (nrnran123_State**)(&_p_rng);\n"
  "    nrnran123_getids3(*pv, di, di+1, di+2);\n"
  "\n"
  "    char which;\n"
  "    nrnran123_getseq(*pv, di+3, &which);\n"
  "    di[4] = (int)which;\n"
  "    //printf(\"ProbAMPANMDA_EMS bbcore_write %d %d %d\\n\", di[0], di[1], di[2]);\n"
  "    }\n"
  "  *offset += 5;\n"
  "}\n"
  "\n"
  "static void bbcore_read(double* x, int* d, int* xx, int* offset, _threadargsproto_) {\n"
  "  assert(!_p_rng);\n"
  "  uint32_t* di = ((uint32_t*)d) + *offset;\n"
  "  if (di[0] != 0 || di[1] != 0 || di[2] != 0) {\n"
  "      nrnran123_State** pv = (nrnran123_State**)(&_p_rng);\n"
  "      *pv = nrnran123_newstream3(di[0], di[1], di[2]);\n"
  "      unsigned char which = (unsigned char)di[4];\n"
  "      nrnran123_setseq(*pv, di[3], which);\n"
  "  }\n"
  "  //printf(\"ProbAMPANMDA_EMS bbcore_read %d %d %d\\n\", di[0], di[1], di[2]);\n"
  "  *offset += 5;\n"
  "}\n"
  "ENDVERBATIM\n"
  "\n"
  ;
#endif
