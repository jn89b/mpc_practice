/* This file was automatically generated by CasADi 3.6.1.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) toy_car_dynamics_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s1[6] = {2, 1, 0, 2, 0, 1};

/* toy_car_dynamics:(states[3],controls[2])->(z_dot[3]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real w0, w1, w2;
  /* #0: @0 = input[1][0] */
  w0 = arg[1] ? arg[1][0] : 0;
  /* #1: @1 = input[0][2] */
  w1 = arg[0] ? arg[0][2] : 0;
  /* #2: @2 = cos(@1) */
  w2 = cos( w1 );
  /* #3: @2 = (@0*@2) */
  w2  = (w0*w2);
  /* #4: output[0][0] = @2 */
  if (res[0]) res[0][0] = w2;
  /* #5: @1 = sin(@1) */
  w1 = sin( w1 );
  /* #6: @0 = (@0*@1) */
  w0 *= w1;
  /* #7: output[0][1] = @0 */
  if (res[0]) res[0][1] = w0;
  /* #8: @0 = input[1][1] */
  w0 = arg[1] ? arg[1][1] : 0;
  /* #9: output[0][2] = @0 */
  if (res[0]) res[0][2] = w0;
  return 0;
}

CASADI_SYMBOL_EXPORT int toy_car_dynamics(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int toy_car_dynamics_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int toy_car_dynamics_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void toy_car_dynamics_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int toy_car_dynamics_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void toy_car_dynamics_release(int mem) {
}

CASADI_SYMBOL_EXPORT void toy_car_dynamics_incref(void) {
}

CASADI_SYMBOL_EXPORT void toy_car_dynamics_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int toy_car_dynamics_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int toy_car_dynamics_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real toy_car_dynamics_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* toy_car_dynamics_name_in(casadi_int i) {
  switch (i) {
    case 0: return "states";
    case 1: return "controls";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* toy_car_dynamics_name_out(casadi_int i) {
  switch (i) {
    case 0: return "z_dot";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* toy_car_dynamics_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* toy_car_dynamics_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int toy_car_dynamics_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 4;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 3;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
