#pragma once

#include "share/atm_process/atmosphere_diagnostic.hpp"

namespace scream {

// Extracts zonal (U) or meridional (V) wind from horiz_winds
// at a specified height above surface or sea level.
//
// Supported field names:
//   U_at_<Y>m_above_surface   V_at_<Y>m_above_surface
//   U_at_<Y>m_above_sealevel  V_at_<Y>m_above_sealevel
//
class HorizWindsAtHeight : public AtmosphereDiagnostic {
public:
  HorizWindsAtHeight (const ekat::Comm& comm, const ekat::ParameterList& params);

  std::string name () const override { return "HorizWindsAtHeight"; }

  void set_grids (const std::shared_ptr<const GridsManager> grids_manager) override;

protected:
  void initialize_impl (const RunType run_type) override;
  void run_impl        (const double dt) override;
  void finalize_impl   () override {};

  Real        m_height;    // target height [m]
  std::string m_surf;      // "surface" or "sealevel"
  int         m_comp_idx;  // 0 = U (zonal), 1 = V (meridional)
  int         m_num_cols;
  int         m_num_levs;
};

} // namespace scream
