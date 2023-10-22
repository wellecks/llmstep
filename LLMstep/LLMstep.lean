/-
`llmstep` tactic for LLM-based next-step suggestions in Lean4.
Examples:
 llmstep ""
 llmstep "have"
 llmstep "apply Continuous"

Author: Sean Welleck
-/
import Lean.Widget.UserWidget
import Std.Lean.Position
import Std.Lean.Format
import Std.Data.String.Basic

open Lean

/- Calls a `suggest.py` python script with the given prefix and pretty-printed goal. -/
def runSuggestAux (path goal pre: String) : IO (List String) := do
  let s ← IO.Process.run { cmd := "python3", args := #[path, goal, pre] }
  return s.splitOn "[SUGGESTION]"

def runSuggest (pre goal : String) : IO (List String) := do
  let cwd ← IO.currentDir
  let path := cwd / "python" / "suggest.py"
  match ← path.pathExists with
  | true => runSuggestAux path.toString goal pre
  | false => do
    let path := cwd / "lake-packages" / "llmstep" / "python" / "suggest.py"
    match ← path.pathExists with
    | true => runSuggestAux path.toString goal pre
    | false => do
      dbg_trace f!"{path}"
      throw <| IO.userError "could not find python script suggest.py"

/- Display clickable suggestions in the VSCode Lean Infoview.
    When a suggestion is clicked, this widget replaces the `llmstep` call
    with the suggestion, and saves the call in an adjacent comment.
    Code based on `Std.Tactic.TryThis.tryThisWidget`. -/
@[widget] def llmstepTryThisWidget : Widget.UserWidgetDefinition where
  name := "llmstep suggestions"
  javascript := "
import * as React from 'react';
import { EditorContext } from '@leanprover/infoview';
const e = React.createElement;
export default function(props) {
  const editorConnection = React.useContext(EditorContext)
  function onClick(suggestion) {
    editorConnection.api.applyEdit({
      changes: { [props.pos.uri]: [{ range:
        props.range,
        newText: suggestion[0] + ' -- ' + props.tactic
        }] }
    })
  }
  return e('div',
  {className: 'ml1'},
  e('ul', {className: 'font-code pre-wrap'}, [
    'Try this: ',
    ...(props.suggestions.map((suggestion, i) =>
        e('li', {onClick: () => onClick(suggestion),
          className:
            suggestion[1] === 'ProofDone' ? 'link pointer dim green' :
            suggestion[1] === 'Valid' ? 'link pointer dim blue' :
            'link pointer dim',
          title: 'Apply suggestion'},
          suggestion[1] === 'ProofDone' ? '🎉 ' + suggestion[0] : suggestion[0]
      )
    )),
    props.info
  ]))
}"


inductive CheckResult : Type
  | ProofDone
  | Valid
  | Invalid
  deriving ToJson, Ord

/- Check whether the suggestion `s` completes the proof, is valid (does
not result in an error message), or is invalid. -/
def checkSuggestion (s: String) : Lean.Elab.Tactic.TacticM CheckResult := do
  withoutModifyingState do
  try
    match Parser.runParserCategory (← getEnv) `tactic s with
      | Except.ok stx =>
        try
          _ ← Lean.Elab.Tactic.evalTactic stx
          let goals ← Lean.Elab.Tactic.getUnsolvedGoals
          if (← getThe Core.State).messages.hasErrors then
            pure CheckResult.Invalid
          else if goals.isEmpty then
            pure CheckResult.ProofDone
          else
            pure CheckResult.Valid
        catch _ =>
          pure CheckResult.Invalid
      | Except.error _ =>
        pure CheckResult.Invalid
    catch _ => pure CheckResult.Invalid


/- Adds multiple suggestions to the Lean InfoView.
   Code based on `Std.Tactic.addSuggestion`. -/
def addSuggestions (tacRef : Syntax) (pfxRef: Syntax) (suggestions: List String)
    (origSpan? : Option Syntax := none)
    (extraMsg : String := "") : Lean.Elab.Tactic.TacticM Unit := do
  if let some tacticRange := (origSpan?.getD tacRef).getRange? then
    if let some argRange := (origSpan?.getD pfxRef).getRange? then
      let map ← getFileMap
      let start := findLineStart map.source tacticRange.start
      let body := map.source.findAux (· ≠ ' ') tacticRange.start start

      let checks ← suggestions.mapM checkSuggestion
      let texts := suggestions.map fun text => (
        (Std.Format.prettyExtra (text.stripSuffix "\n")
         (indent := (body - start).1)
         (column := (tacticRange.start - start).1)
      ))

      let textsAndChecks := texts.zip checks |>.toArray |>.qsort
        fun a b => compare a.2 b.2 = Ordering.lt

      let start := (tacRef.getRange?.getD tacticRange).start
      let stop := (pfxRef.getRange?.getD argRange).stop
      let stxRange :=
      { start := map.lineStart (map.toPosition start).line
        stop := map.lineStart ((map.toPosition stop).line + 1) }
      let full_range : String.Range :=
      { start := tacticRange.start, stop := argRange.stop }
      let full_range := map.utf8RangeToLspRange full_range
      let tactic := Std.Format.prettyExtra f!"{tacRef.prettyPrint}{pfxRef.prettyPrint}"
      let json := Json.mkObj [
        ("tactic", tactic),
        ("suggestions", toJson textsAndChecks),
        ("range", toJson full_range),
        ("info", extraMsg)
      ]
      Widget.saveWidgetInfo ``llmstepTryThisWidget json (.ofRange stxRange)

/--
Call the LLM on a goal, asking for suggestions beginning with a prefix.
-/
def llmStep (pre : String) (g : MVarId) : MetaM (List String) := do
  let pp := toString (← Meta.ppGoal g)
  runSuggest pre pp

open Lean Elab Tactic

/- `llmstep` tactic.
   Examples:
    llmstep ""
    llmstep "have"
    llmstep "apply Continuous" -/
syntax "llmstep" str: tactic
elab_rules : tactic
  | `(tactic | llmstep%$tac $pfx:str) => do
    addSuggestions tac pfx (← liftMetaMAtMain (llmStep pfx.getString))
