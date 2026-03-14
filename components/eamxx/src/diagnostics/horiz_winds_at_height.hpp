#ifndef EAMXX_HORIZ_WINDS_AT_HEIGHT_HPP
#define EAMXX_HORIZ_WINDS_AT_HEIGHT_HPP

#include "share/atm_process/atmosphere_diagnostic.hpp"

namespace scream
{

/*
 * Diagnostic that extracts a single horizontal wind component (zonal U or
 * meridional V) from horiz_winds and interpolates it to a given height above
 * the surface or sea level.
 *
 * Constructed via scorpio when the output YAML requests e.g.:
 *   U_at_10m_above_surface   (zonal wind at 10 m above surface)
 *   V_at_10m_above_sealevel  (meridional wind at 10 m above sea level)
 *
 * Parameters (set by AtmosphereOutput::create_diagnostic):
 *   wind_component    (int)    : 0 = U (zonal), 1 = V (meridional)
 *   surface_reference (string) : "surface" or "sealevel"
 *   vertical_location (string) : e.g. "10m"
 *   grid_name         (string) : name of the output grid
 */
class HorizWindsAtHeight : public AtmosphereDiagnostic
{
public:
  HorizWindsAtHeight (const ekat::Comm& comm, const ekat::ParameterList& params);

  // Returns e.g. "U_at_10m_above_surface" — matches the requested field name
  std::string name () const { return m_diag_name; }

  void set_grids (const std::shared_ptr<const GridsManager> grids_manager);

protected:
#ifdef KOKKOS_ENABLE_CUDA
public:
#endif
  void compute_diagnostic_impl ();
protected:
  void initialize_impl (const RunType /*run_type*/);

  std::string  m_diag_name;   // e.g. "U_at_10m_above_surface"
  std::string  m_z_name;      // "z" (sealevel) or "height" (surface)
  std::string  m_z_suffix;    // "_mid" or "_int" (set in initialize_impl)
  int          m_comp_idx;    // 0 = U (horiz_winds[:,0,:]), 1 = V ([:,1,:])
  Real         m_z;           // target height in metres
};

} //namespace scream

#endif // EAMXX_HORIZ_WINDS_AT_HEIGHT_HPP
