import LLMstep
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Analysis.Convex.Jensen
import Mathlib.Algebra.Order.Field.Basic

noncomputable section

open Classical BigOperators Real

variable {Ω : Type*}[Fintype Ω]

-- My own probability mass function
structure my_pmf (Ω : Type*)[Fintype Ω] :=
  (f : Ω → ℝ)
  (non_neg : ∀ x : Ω, 0 ≤ f x)
  (sum_one : HasSum f 1)

instance funLike : FunLike (my_pmf Ω) Ω fun _ => ℝ where
  coe p x := p.f x
  coe_injective' p q h := by
    cases p
    cases q
    congr

theorem hasSum_sum_one (p : my_pmf Ω) : HasSum p 1 := p.sum_one

-- Probability of any outcome is at most one.
theorem px_le_one (p : my_pmf Ω) (x : Ω) : p x ≤ 1 := by
  refine' hasSum_le _ (hasSum_ite_eq x (p x)) (hasSum_sum_one p)
  intro x
  split_ifs with h
  rw [h]
  exact p.non_neg _
