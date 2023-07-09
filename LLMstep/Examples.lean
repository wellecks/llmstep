import LLMstep

import Mathlib.Data.Real.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Topology.Instances.Real
import Mathlib.Data.Nat.Prime
import Mathlib.Algebra.BigOperators.Order


/- Examples -/
example : 2 = 2 := by
  rfl -- llmstep "" 


example (f : ℕ → ℕ) : Monotone f → ∀ n, f n ≤ f (n + 1) := by
  intro h n -- llmstep "" 
  apply h -- llmstep "apply" 
  apply Nat.le_succ -- llmstep "apply" 


-- Proving the example above in a different way.
example (f : ℕ → ℕ) : Monotone f → ∀ n, f n ≤ f (n + 1) := by
  intro h n
  exact h (Nat.le_succ _) -- llmstep "exact" 


example {α : Type _} (r s t : Set α) : r ⊆ s → s ⊆ t → r ⊆ t := by
  intro hrs hst x hxr
  apply hst -- llmstep "apply" 
  apply hrs hxr -- llmstep "apply" 


-- Proving the example above in a different way.
example {α : Type _} (r s t : Set α) : r ⊆ s → s ⊆ t → r ⊆ t := by
  intro hrs hst x hxr
  have hxs : x ∈ s := hrs hxr -- llmstep "have" 
  have hxt : x ∈ t := hst hxs -- llmstep "have" 
  exact hxt -- llmstep "" 


-- Example from Mathematics in Lean C08
example {X Y : Type _} [MetricSpace X] [MetricSpace Y] {f : X → Y} (hf : Continuous f) :
    Continuous fun p : X × X ↦ dist (f p.1) (f p.2) := by
    apply Continuous.dist -- llmstep "apply Continuous" 
    exact hf.comp continuous_fst -- llmstep "" 
    exact hf.comp continuous_snd -- llmstep "" 

