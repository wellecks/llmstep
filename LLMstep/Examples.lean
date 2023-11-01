/- Examples -/

import LLMstep

import Mathlib.Data.Real.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Topology.Instances.Real
import Mathlib.Data.Nat.Prime
import Mathlib.Algebra.BigOperators.Order

/-
First, here is a trivial example.
We would start this proof using:

  example : 2 = 2 := by
   llmstep ""

then choose a suggestion. Clicking the suggestion
would yield the proof below:
-/
example : 2 = 2 := by
  rfl

/-
As shown in the rest of the examples below, we can use the
llmstep tactic again on later steps and with other prefixes.
Naturally, steps that use llmstep can be interleaved with steps
that do not use llmstep.
-/

example (f : ℕ → ℕ) : Monotone f → ∀ n, f n ≤ f (n + 1) := by
  -- llmstep ""
  intro h n
  -- llmstep "apply"
  apply h
  -- llmstep "apply"
  apply Nat.le_succ


-- Proving the example above in a different way.
example (f : ℕ → ℕ) : Monotone f → ∀ n, f n ≤ f (n + 1) := by
  intro h n
  -- llmstep "exact"
  exact h (Nat.le_succ _)


example {α : Type _} (r s t : Set α) : r ⊆ s → s ⊆ t → r ⊆ t := by
  intro hrs hst x hxr
  -- llmstep "apply"
  apply hst
  -- llmstep "apply"
  apply hrs hxr


-- Proving the example above in a different way.
example {α : Type _} (r s t : Set α) : r ⊆ s → s ⊆ t → r ⊆ t := by
  intro hrs hst x hxr
  -- llmstep "have"
  have hxs : x ∈ s := hrs hxr
  -- llmstep "have"
  have hxt : x ∈ t := hst hxs
  -- llmstep ""
  exact hxt

-- Example from Mathematics in Lean C08
example {X Y : Type _} [MetricSpace X] [MetricSpace Y] {f : X → Y} (hf : Continuous f) :
    Continuous fun p : X × X ↦ dist (f p.1) (f p.2) := by
    -- llmstep ""
    exact continuous_dist.comp (hf.prod_map hf)


-- Example from ProofNet (Rudin)
theorem exercise_1_18b : ¬ ∀ (x : ℝ), ∃ (y : ℝ), y ≠ 0 ∧ x * y = 0 := by
  -- llmstep ""
  push_neg
  -- llmstep ""
  use 1
  -- llmstep ""
  simp [ne_of_gt]

/- Using context

The example below involves a theorem about the newly defined `my_object`.
Llmstep with a model that leverages document context (e.g. llemma)
can give suggestions that use properties of `my_object`:
-/
variable {Ω : Type*}[Fintype Ω]

structure my_object (Ω : Type*)[Fintype Ω] :=
  (f : Ω → ℝ)
  (cool_property : ∀ x : Ω, 0 ≤ f x)

theorem my_object_sum_nonneg (o1 o2: my_object Ω) : o1.f + o2.f ≥ 0 := by
  -- llmstep "" (with llemma)
  apply add_nonneg
  -- llmstep "" (with llemma)
  · apply o1.cool_property
  -- llmstep "" (with llemma)
  · apply o2.cool_property
