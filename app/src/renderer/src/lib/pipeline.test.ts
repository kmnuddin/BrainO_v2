import { describe, expect, it } from "vitest";
import type { PipelineSpec } from "../engine/types";
import { emptySpec, inputOf, moved, newStepId, problemsByStep, sameSpec } from "./pipeline";

function spec(): PipelineSpec {
  return {
    ...emptySpec("oddball"),
    steps: [
      { id: "preprocess", tool: "eeg.preprocess", args: {} },
      { id: "ica", tool: "eeg.ica_clean", args: {} },
      { id: "epochs", tool: "eeg.epoch", args: {} },
      { id: "erp", tool: "eeg.erp", args: {} },
    ],
    group: [{ id: "test", tool: "stats.cluster_permutation", args: { data: "erp" } }],
  };
}

describe("inputOf", () => {
  it("follows the engine's order rules", () => {
    const s = spec();
    expect(inputOf(s, s.steps[0], false)).toEqual({ step: null });
    expect(inputOf(s, s.steps[1], false).step?.id).toBe("preprocess");
    expect(inputOf(s, s.steps[2], false).step?.id).toBe("ica");
    expect(inputOf(s, s.steps[3], false).step?.id).toBe("epochs");
    expect(inputOf(s, s.group[0], true).step?.id).toBe("erp");
  });

  it("names the missing step", () => {
    const s = spec();
    s.steps = [s.steps[3]];
    expect(inputOf(s, s.steps[0], false)).toEqual({ step: null, needs: "eeg.epoch" });
    s.group[0].args = { data: "psd" };
    expect(inputOf(s, s.group[0], true)).toEqual({ step: null, needs: "eeg.spectral" });
  });
});

describe("newStepId", () => {
  it("numbers repeated tools", () => {
    expect(newStepId(spec(), "eeg.spectral")).toBe("spectral");
    expect(newStepId(spec(), "eeg.preprocess")).toBe("preprocess_2");
    expect(newStepId(spec(), "stats.cluster_permutation")).toBe("cluster_permutation");
  });
});

describe("problemsByStep", () => {
  it("splits the engine's messages by step and setting", () => {
    const problems = problemsByStep({
      valid: false,
      problems: [
        "step 'preprocess' (eeg.preprocess): h_freq: Input should be greater than 0; l_freq: bad",
        "group step 'test': stats.cluster_permutation works on the group",
        "step ids must be unique; repeated: erp",
      ],
      findings: [
        { severity: "error", step: "erp", message: "eeg.erp reads epochs: put eeg.epoch before it" },
        { severity: "warning", step: null, message: "only 2 subjects" },
      ],
    });
    expect(problems.get("preprocess")?.fields).toEqual({ h_freq: "Input should be greater than 0", l_freq: "bad" });
    expect(problems.get("test")?.messages).toEqual(["stats.cluster_permutation works on the group"]);
    expect(problems.get("erp")?.messages).toEqual(["eeg.erp reads epochs: put eeg.epoch before it"]);
    expect(problems.get("")?.messages).toEqual(["step ids must be unique; repeated: erp", "Warning: only 2 subjects"]);
  });
});

describe("helpers", () => {
  it("moves within bounds", () => {
    expect(moved([1, 2, 3], 0, 1)).toEqual([2, 1, 3]);
    expect(moved([1, 2, 3], 0, -1)).toEqual([1, 2, 3]);
  });

  it("compares specs whatever the key order", () => {
    const a = spec();
    const b = JSON.parse(JSON.stringify(a)) as PipelineSpec;
    b.steps[0].args = { b: 1, a: 2 };
    a.steps[0].args = { a: 2, b: 1 };
    expect(sameSpec(a, b)).toBe(true);
    b.steps[0].args = { a: 3 };
    expect(sameSpec(a, b)).toBe(false);
  });
});
