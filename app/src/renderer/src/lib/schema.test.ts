import { describe, expect, it } from "vitest";
import type { JsonSchema } from "../engine/types";
import { describeDefault, fields, parseNumber, textToList, withArg } from "./schema";

// Shapes taken from the engine's tool schemas (Pydantic's JSON schema).
const EPOCH: JsonSchema = {
  type: "object",
  required: ["subject", "conditions"],
  properties: {
    subject: { type: "string", title: "Subject" },
    task: { anyOf: [{ type: "string" }, { type: "null" }], default: null, title: "Task" },
    conditions: {
      type: "object",
      additionalProperties: { type: "array", items: { type: "string" } },
      title: "Conditions",
    },
    tmin: { type: "number", default: -0.2, title: "Tmin" },
    seed: { type: "integer", default: 0 },
    baseline: {
      anyOf: [
        {
          type: "array",
          minItems: 2,
          maxItems: 2,
          prefixItems: [
            { anyOf: [{ type: "number" }, { type: "null" }] },
            { anyOf: [{ type: "number" }, { type: "null" }] },
          ],
        },
        { type: "null" },
      ],
      default: [null, 0],
    },
    reject: { type: "string", enum: ["autoreject", "none"], default: "autoreject" },
    reference: {
      anyOf: [{ type: "string", enum: ["average", "none"] }, { type: "array", items: { type: "string" } }],
      default: "average",
    },
    notch_freqs: { type: "array", items: { type: "number" }, default: [] },
    bands: {
      type: "object",
      additionalProperties: {
        type: "array",
        prefixItems: [{ type: "number" }, { type: "number" }],
        minItems: 2,
        maxItems: 2,
      },
    },
    measures: { type: "array", items: { $ref: "#/$defs/Measure" } },
    n_components: { anyOf: [{ type: "integer" }, { type: "number" }, { type: "null" }], default: null },
    weird: { type: "object" },
  },
  $defs: {
    Measure: {
      type: "object",
      properties: { name: { type: "string" }, tmin: { type: "number" } },
      required: ["name", "tmin"],
    },
  },
};

describe("fields", () => {
  const byName = Object.fromEntries(fields(EPOCH, ["subject"]).map((f) => [f.name, f]));

  it("leaves out hidden fields and keeps the order", () => {
    expect(Object.keys(byName)[0]).toBe("task");
    expect(byName.subject).toBeUndefined();
  });

  it("reads each shape", () => {
    expect(byName.task.kind).toEqual({ kind: "text" });
    expect(byName.task.nullable).toBe(true);
    expect(byName.conditions.kind).toEqual({ kind: "map-list" });
    expect(byName.conditions.required).toBe(true);
    expect(byName.tmin.kind).toEqual({ kind: "number", integer: false });
    expect(byName.seed.kind).toEqual({ kind: "number", integer: true });
    expect(byName.baseline.kind).toEqual({ kind: "tuple", size: 2, nullable: [true, true] });
    expect(byName.baseline.nullable).toBe(true);
    expect(byName.reject.kind).toEqual({ kind: "enum", options: ["autoreject", "none"] });
    expect(byName.reference.kind).toEqual({ kind: "enum-or-list", options: ["average", "none"] });
    expect(byName.notch_freqs.kind).toEqual({ kind: "list", item: "number" });
    expect(byName.bands.kind).toEqual({ kind: "map-tuple", size: 2 });
    expect(byName.measures.kind.kind).toBe("objects");
    expect(byName.n_components.kind).toEqual({ kind: "number", integer: false });
    expect(byName.weird.kind).toEqual({ kind: "json" });
  });

  it("describes defaults", () => {
    expect(describeDefault(byName.baseline)).toBe("[start, 0]");
    expect(describeDefault(byName.task)).toBe("not set");
    expect(describeDefault(byName.conditions)).toBe("required");
    expect(describeDefault(byName.notch_freqs)).toBe("none");
  });
});

describe("values", () => {
  it("parses lists and numbers", () => {
    expect(textToList(" Fz, Cz ,,Pz ")).toEqual(["Fz", "Cz", "Pz"]);
    expect(textToList("50, 100", true)).toEqual([50, 100]);
    expect(textToList("50, x", true)).toBeNull();
    expect(parseNumber("", false)).toBeNull();
    expect(parseNumber("0.5", false)).toBe(0.5);
    expect(parseNumber("0.5", true)).toBeUndefined();
    expect(parseNumber("abc", false)).toBeUndefined();
  });

  it("sets and removes arguments", () => {
    expect(withArg({ a: 1 }, "b", 2)).toEqual({ a: 1, b: 2 });
    expect(withArg({ a: 1, b: 2 }, "b", undefined)).toEqual({ a: 1 });
  });
});
