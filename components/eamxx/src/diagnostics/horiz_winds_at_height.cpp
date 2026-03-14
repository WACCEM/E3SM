#include "diagnostics/horiz_winds_at_height.hpp"

#include "ekat/std_meta/ekat_std_utils.hpp"
#include "ekat/util/ekat_units.hpp"

namespace
{
// Reuse the same binary-search helper as FieldAtHeight:
// find the first position in [beg,end) whose value is smaller than z.
template<typename T>
KOKKOS_INLINE_FUNCTION
const T* find_first_smaller_z (const T* beg, const T* end, const T& z)
{
  int count = end - beg;
  while (count>1) {
    auto mid = beg + count/2 - 1;
    if (*mid>=z) {
      beg = mid+1;
    } else {
      end = mid+1;
    }
    count = end - beg;
  }
  return *beg < z ? beg : end;
}
} // anonymous namespace

namespace scream
{

HorizWindsAtHeight::
HorizWindsAtHeight (const ekat::Comm& comm, const ekat::ParameterList& params)
 : AtmosphereDiagnostic(comm,params)
{
  // wind_component: 0 = U (zonal), 1 = V (meridional)
  m_comp_idx = m_params.get<int>("wind_component");

  const auto surf_ref = m_params.get<std::string>("surface_reference");
  EKAT_REQUIRE_MSG(surf_ref=="sealevel" or surf_ref=="surface",
      "Error! Invalid surface reference for HorizWindsAtHeight.\n"
      " - surface reference: " + surf_ref + "\n"
      " -     valid options: sealevel, surface\n");
  m_z_name = (surf_ref=="sealevel") ? "z" : "height";

  const auto& location = m_params.get<std::string>("vertical_location");
  auto chars_start = location.find_first_not_of("0123456789.");
  EKAT_REQUIRE_MSG (chars_start!=0 && chars_start!=std::string::npos,
      "Error! Invalid string for height value for HorizWindsAtHeight.\n"
      " - input string   : " + location + "\n"
      " - expected format: Nm, with N a number\n");
  m_z = std::stod(location.substr(0,chars_start));

  const auto wind_name = (m_comp_idx==0) ? "U" : "V";
  m_diag_name = std::string(wind_name) + "_at_"
              + m_params.get<std::string>("vertical_location")
              + "_above_" + surf_ref;
}

void HorizWindsAtHeight::
set_grids (const std::shared_ptr<const GridsManager> grids_manager)
{
  const auto& gname = m_params.get<std::string>("grid_name");

  // horiz_winds and z fields are looked up on the requested grid.
  // The framework will handle any necessary remapping.
  add_field<Required>("horiz_winds", gname);
  add_field<Required>(m_z_name+"_mid", gname);
  add_field<Required>(m_z_name+"_int", gname);
}

void HorizWindsAtHeight::
initialize_impl (const RunType /*run_type*/)
{
  const auto& f   = get_field_in("horiz_winds");
  const auto& fid = f.get_header().get_identifier();

  using namespace ShortFieldTagsNames;
  const auto& layout = fid.get_layout();
  const auto tag = layout.tags().back();
  EKAT_REQUIRE_MSG (tag==LEV || tag==ILEV,
      "Error! HorizWindsAtHeight: horiz_winds must have LEV or ILEV as last tag.\n"
      " - field layout: " + layout.to_string() + "\n");
  m_z_suffix = (tag==LEV) ? "_mid" : "_int";

  // horiz_winds layout is (CMP, LEV). Strip both CMP and LEV to get a scalar
  // layout per column. With the distributed COL dimension the total field is
  // 1-D (ncols,).
  auto out_layout = layout.clone().strip_dims({tag, CMP});
  FieldIdentifier d_fid(m_diag_name, out_layout,
                        fid.get_units(), fid.get_grid_name());
  m_diagnostic_output = Field(d_fid);
  m_diagnostic_output.allocate_view();

  // Propagate any io string attributes from the input field
  using stratts_t = std::map<std::string,std::string>;
  const auto& src = get_fields_in().front();
  const auto& src_atts =
      src.get_header().get_extra_data<stratts_t>("io: string attributes");
  auto& dst_atts =
      m_diagnostic_output.get_header().get_extra_data<stratts_t>("io: string attributes");
  for (const auto& [name, val] : src_atts) {
    dst_atts[name] = val;
  }
}

// =============================================================================
void HorizWindsAtHeight::compute_diagnostic_impl()
{
  const auto z_view  = get_field_in(m_z_name + m_z_suffix).get_view<const Real**>();
  const auto hw_view = get_field_in("horiz_winds").get_view<const Real***>();
  const auto d_view  = m_diagnostic_output.get_view<Real*>();

  using RangePolicy = typename KokkosTypes<DefaultDevice>::RangePolicy;

  const auto z_tgt = m_z;
  const auto comp  = m_comp_idx;
  const int  ncols = hw_view.extent(0);
  const int  nlevs = hw_view.extent(2);

  RangePolicy policy (0, ncols);
  Kokkos::parallel_for(policy,
      KOKKOS_LAMBDA(const int i) {
        auto z_i = ekat::subview(z_view, i);

        auto beg = z_i.data();
        auto end = beg + nlevs;
        auto it  = find_first_smaller_z(beg, end, z_tgt);

        if (it==beg) {
          // Target is above all levels: extrapolate with top value
          d_view(i) = hw_view(i, comp, 0);
        } else if (it==end) {
          // Target is below all levels: extrapolate with bottom value
          d_view(i) = hw_view(i, comp, nlevs-1);
        } else {
          auto pos = it - beg;
          auto z0  = z_i(pos-1);
          auto z1  = z_i(pos);
          auto f0  = hw_view(i, comp, pos-1);
          auto f1  = hw_view(i, comp, pos);
          d_view(i) = ( (z_tgt-z0)*f1 + (z1-z_tgt)*f0 ) / (z1-z0);
        }
  });
}

} //namespace scream

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
