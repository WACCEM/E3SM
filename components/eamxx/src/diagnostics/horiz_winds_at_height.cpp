#include "diagnostics/horiz_winds_at_height.hpp"
#include <ekat/ekat_assert.hpp>

namespace scream {

HorizWindsAtHeight::HorizWindsAtHeight (const ekat::Comm& comm,
                                        const ekat::ParameterList& params)
  : AtmosphereDiagnostic(comm, params)
{
  const auto& name = params.get<std::string>("diag_name");

  EKAT_REQUIRE_MSG (name.size() > 2 && (name[0]=='U' || name[0]=='V') && name[1]=='_',
      "Error! HorizWindsAtHeight: diag name must start with 'U_' or 'V_'.\n"
      "  Actual name: " + name + "\n");
  m_comp_idx = (name[0] == 'U') ? 0 : 1;

  auto pos = name.rfind("_at_");
  EKAT_REQUIRE_MSG (pos != std::string::npos,
      "Error! HorizWindsAtHeight: expected format [U|V]_at_<val>m_above_[surface|sealevel]\n"
      "  Actual name: " + name + "\n");
  auto suffix = name.substr(pos + 4);

  auto pos2 = suffix.find("m_above_");
  EKAT_REQUIRE_MSG (pos2 != std::string::npos,
      "Error! HorizWindsAtHeight: expected format [U|V]_at_<val>m_above_[surface|sealevel]\n"
      "  Actual name: " + name + "\n");
  m_height = std::stof(suffix.substr(0, pos2));
  m_surf   = suffix.substr(pos2 + 8);

  EKAT_REQUIRE_MSG (m_surf == "surface" || m_surf == "sealevel",
      "Error! HorizWindsAtHeight: reference surface must be 'surface' or 'sealevel'.\n"
      "  Actual value: " + m_surf + "\n");
}

void HorizWindsAtHeight::set_grids (
    const std::shared_ptr<const GridsManager> grids_manager)
{
  const auto& name = params().get<std::string>("diag_name");
  const auto& gn   = params().get<std::string>("grid_name");

  auto grid = grids_manager->get_grid("Physics GLL");
  m_num_cols = grid->get_num_local_dofs();
  m_num_levs = grid->get_num_vertical_levels();

  add_field<Required>("horiz_winds",
                      FieldLayout({CMP, LEV}, {2, m_num_levs}),
                      m/s, "Physics GLL");

  add_field<Required>("z_mid",
                      FieldLayout({COL, LEV}, {m_num_cols, m_num_levs}),
                      ekat::units::m, gn);

  FieldLayout diag_layout ({COL}, {m_num_cols});
  FieldIdentifier fid(name, diag_layout, m/s, gn);
  m_diagnostic_output = Field(fid);
  m_diagnostic_output.allocate_view();
}

void HorizWindsAtHeight::initialize_impl (const RunType /*run_type*/) {}

void HorizWindsAtHeight::run_impl (const double /*dt*/)
{
  const auto hw_v  = get_field_in("horiz_winds").get_view<const Real***>();
  const auto z_v   = get_field_in("z_mid").get_view<const Real**>();
  const auto out_v = m_diagnostic_output.get_view<Real*>();

  const int  nlev   = m_num_levs;
  const int  comp   = m_comp_idx;
  const Real height = m_height;
  const std::string surf = m_surf;

  Kokkos::parallel_for(m_num_cols, KOKKOS_LAMBDA(const int icol) {
    int  k_above = -1;
    Real z_above = 0, z_below = 0;

    for (int k = nlev - 1; k >= 0; --k) {
      const Real z_val = (surf == "surface")
                         ? (z_v(icol,k) - z_v(icol,nlev-1))
                         : z_v(icol,k);
      if (z_val >= height) {
        k_above = k;
        z_above = z_val;
        z_below = (k < nlev-1)
                  ? ((surf == "surface")
                     ? (z_v(icol,k+1) - z_v(icol,nlev-1))
                     : z_v(icol,k+1))
                  : 0.0;
      } else {
        break;
      }
    }

    if (k_above == -1) {
      out_v(icol) = hw_v(icol, comp, nlev-1);
    } else if (k_above == nlev-1 || z_above == z_below) {
      out_v(icol) = hw_v(icol, comp, k_above);
    } else {
      const Real alpha = (height - z_below) / (z_above - z_below);
      out_v(icol) = hw_v(icol, comp, k_above+1)
                  + alpha * (hw_v(icol, comp, k_above)
                           - hw_v(icol, comp, k_above+1));
    }
  });
}

} // namespace scream
