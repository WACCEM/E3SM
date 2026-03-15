#include "diagnostics/field_horiz_avg.hpp"

#include <ekat/ekat_pack_utils.hpp>
#include <ekat/kokkos/ekat_kokkos_utils.hpp>

#include <mpi.h>

namespace scream
{

FieldHorizAvgDiagnostic::
FieldHorizAvgDiagnostic (const ekat::Comm& comm, const ekat::ParameterList& params)
  : AtmosphereDiagnostic(comm, params)
{
  EKAT_REQUIRE_MSG(m_params.isParameter("field_name"),
      "Error! FieldHorizAvgDiagnostic requires 'field_name' in parameters.\n");
  m_fname     = m_params.get<std::string>("field_name");
  m_diag_name = m_fname + "_horiz_avg";
}

void FieldHorizAvgDiagnostic::
set_grids (const std::shared_ptr<const GridsManager> grids_manager)
{
  const auto& gname = m_params.get<std::string>("grid_name");
  add_field<Required>(m_fname, gname);
}

void FieldHorizAvgDiagnostic::
initialize_impl (const RunType /*run_type*/)
{
  const auto& f      = get_field_in(m_fname);
  const auto& fid    = f.get_header().get_identifier();
  const auto& layout = fid.get_layout();
  const auto& gname  = fid.get_grid_name();

  m_is_3d    = (layout.rank() == 2);
  m_num_cols = layout.dim(0);

  // Compute total columns across all MPI ranks
  int local_ncols = m_num_cols;
  MPI_Allreduce(MPI_IN_PLACE, &local_ncols, 1, MPI_INT, MPI_SUM, m_comm.mpi_comm());
  m_ncols_global = local_ncols;

  if (m_is_3d) {
    m_num_levs = layout.dim(1);
    m_col_sum  = view_1d("horiz_avg_col_sum", m_num_levs);
  } else {
    m_num_levs = 0;
    m_col_sum  = view_1d("horiz_avg_col_sum", 1);
  }

  // Allocate the output diagnostic field with the same layout as the input
  FieldIdentifier out_fid(m_diag_name, layout.clone(), fid.get_units(), gname);
  m_diagnostic_output = Field(out_fid);
  m_diagnostic_output.allocate_view();
}

void FieldHorizAvgDiagnostic::
compute_diagnostic_impl ()
{
  const auto& f    = get_field_in(m_fname);
  const int ncols  = m_num_cols;
  const int ncols_g = m_ncols_global;
  auto col_sum     = m_col_sum;

  if (m_is_3d) {
    const int nlevs  = m_num_levs;
    const auto in_v  = f.get_view<const Real**>();
    const auto out_v = m_diagnostic_output.get_view<Real**>();

    // Step 1: Sum over local columns for each vertical level
    Kokkos::parallel_for("FieldHorizAvg::local_sum_3d",
        nlevs,
        KOKKOS_LAMBDA(int jlev) {
          Real s = 0;
          for (int icol = 0; icol < ncols; ++icol) s += in_v(icol, jlev);
          col_sum(jlev) = s;
        });
    Kokkos::fence();

    // Step 2: Copy to host, MPI allreduce sum across ranks, divide by global ncols
    auto h_sum = Kokkos::create_mirror_view(col_sum);
    Kokkos::deep_copy(h_sum, col_sum);
    MPI_Allreduce(MPI_IN_PLACE, h_sum.data(), nlevs,
                  ekat::get_mpi_type<Real>(), MPI_SUM, m_comm.mpi_comm());
    for (int jlev = 0; jlev < nlevs; ++jlev) h_sum(jlev) /= ncols_g;
    Kokkos::deep_copy(col_sum, h_sum);

    // Step 3: Broadcast global mean to all output columns
    Kokkos::parallel_for("FieldHorizAvg::broadcast_3d",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{ncols,nlevs}),
        KOKKOS_LAMBDA(int icol, int jlev) {
          out_v(icol, jlev) = col_sum(jlev);
        });
  } else {
    // 2D surface field: layout (COL)
    const auto in_1d  = f.get_view<const Real*>();
    const auto out_1d = m_diagnostic_output.get_view<Real*>();

    Real local_sum = 0;
    Kokkos::parallel_reduce("FieldHorizAvg::local_sum_2d",
        ncols,
        KOKKOS_LAMBDA(int icol, Real& partial) { partial += in_1d(icol); },
        local_sum);
    Kokkos::fence();
    MPI_Allreduce(MPI_IN_PLACE, &local_sum, 1,
                  ekat::get_mpi_type<Real>(), MPI_SUM, m_comm.mpi_comm());
    local_sum /= ncols_g;
    Kokkos::parallel_for("FieldHorizAvg::broadcast_2d",
        ncols,
        KOKKOS_LAMBDA(int icol) { out_1d(icol) = local_sum; });
  }
}

} // namespace scream
