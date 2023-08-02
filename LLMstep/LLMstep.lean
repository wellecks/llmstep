/-
`llmstep` tactic for LLM-based next-step suggestions in Lean4.
Examples:
 llmstep ""
 llmstep "have"
 llmstep "apply Continuous" 

Author: Sean Welleck
-/
import Mathlib.Tactic

open Lean

/- Calls a `suggest.py` python script with the given `args`. -/
def runSuggest (args : Array String) : IO String := do
  let cwd ← IO.currentDir
  let path := cwd / "python" / "suggest.py"
  unless ← path.pathExists do
    dbg_trace f!"{path}"
    throw <| IO.userError "could not find python script suggest.py"
  let s ← IO.Process.run { cmd := "python3", args := #[path.toString] ++ args }
  return s

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
        newText: suggestion + ' -- ' + props.tactic
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
            props.checks[i] === 'ProofDone' ? 'link pointer dim green' : 
            props.checks[i] === 'Valid' ? 'link pointer dim blue' : 
            'link pointer dim', 
          title: 'Apply suggestion'}, 
          props.checks[i] === 'ProofDone' ? '🎉 ' + suggestion : suggestion
      )
    )),
    props.info
  ]))
}"


inductive CheckResult : Type
  | ProofDone
  | Valid
  | Invalid
  deriving ToJson

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

      let dones := ((texts.zip checks).filter fun (_, check) => match check with
        | CheckResult.ProofDone => true
        | _ => false)

      let valids := ((texts.zip checks).filter fun (_, check) => match check with
        | CheckResult.Valid => true
        | _ => false)

      let invalids := ((texts.zip checks).filter fun (_, check) => match check with
        | CheckResult.Invalid => true
        | _ => False)

      let checks := (dones ++ valids ++ invalids).map fun (_, check) => check
      let texts := (dones ++ valids ++ invalids).map fun (text, _) => text

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
        ("suggestions", toJson texts), 
        ("checks", toJson checks),
        ("range", toJson full_range), 
        ("info", extraMsg)
      ]
      Widget.saveWidgetInfo ``llmstepTryThisWidget json (.ofRange stxRange)


/- `llmstep` tactic.
   Examples:
    llmstep ""
    llmstep "have"
    llmstep "apply Continuous" -/
syntax "llmstep" str: tactic
elab_rules : tactic
  | `(tactic | llmstep%$tac $pfx:str) =>
    Lean.Elab.Tactic.withMainContext do
      let goal ← Lean.Elab.Tactic.getMainGoal
      let ppgoal ← Lean.Meta.ppGoal goal
      let ppgoalstr := toString ppgoal
      let suggest ← runSuggest #[ppgoalstr, pfx.getString]
      addSuggestions tac pfx $ suggest.splitOn "[SUGGESTION]"
  
