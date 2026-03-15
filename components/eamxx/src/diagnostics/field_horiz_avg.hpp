#ifndef EAMXX_FIELD_HORIZ_AVG_DIAGNOSTIC_HPP
#define EAMXX_FIELD_HORIZ_AVG_DIAGNOSTIC_HPP

#include "share/atm_process/atmosphere_diagnostic.hpp"
#include "share/scream_types.hpp"

namespace scream
{

// Computes the horizontal (column) average of any field or diagnostic.
// The output field has the same layout as the input field, with all columns
// holding the global mean value.
//
// Required parameters:
//   - "field_name" : name of the input field or diagnostic
//   - "grid_name"  : name of the grid
class FieldHorizAvgDiagnostic : public AtmosphereDiagnostic
{
public:
  using KT      = KokkosTypes<DefaultDevice>;
  using view_1d = typename KT::template view_1d<Real>;

  FieldHorizAvgDiagnostic (const ekat::Comm& comm, const ekat::ParameterList& params);

  std::string name () const { return m_diag_name; }

  void set_grids (const std::shared_ptr<const GridsManager> grids_manager);

protected:
  void initialize_impl (const RunType run_type);
  void compute_diagnostic_impl ();

  std::string m_fname;       // input field/diagnostic name
  std::string m_diag_name;   // output diagnostic name (= m_fname + "_horiz_avg")

  int  m_num_cols;       // local column count
  int  m_num_levs;       // number of levels (0 for 2D surface fields)
  int  m_ncols_global;   // global column count (across all MPI ranks)
  bool m_is_3d;          // true if field has (COL,LEV) layout; false for (COL)

  // Scratch buffer for per-level sums during horizontal averaging
  view_1d m_col_sum;

}; // class FieldHorizAvgDiagnostic

} // namespace scream

#endif // EAMXX_FIELD_HORIZ_AVG_DIAGNOSTIC_HPP
