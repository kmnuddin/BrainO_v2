/** Reading the tools' JSON schemas (from Pydantic) to build forms.
 *
 * Each property becomes one field of a known kind. The form edits only what the user sets:
 * a field left empty keeps the tool's default, so a pipeline spec stays as short as the YAML a
 * person would write. */

import type { JsonSchema } from "../engine/types";

export type FieldKind =
  | { kind: "text" }
  | { kind: "number"; integer: boolean }
  | { kind: "boolean" }
  | { kind: "enum"; options: string[] }
  /** An option from a list, or a list of names (e.g. reference: "average" or channels). */
  | { kind: "enum-or-list"; options: string[] }
  | { kind: "list"; item: "string" | "number" }
  /** Fixed-length numbers, any of which may be empty (null): e.g. a baseline [null, 0]. */
  | { kind: "tuple"; size: number; nullable: boolean[] }
  /** Name → list of strings, e.g. conditions → event descriptions. */
  | { kind: "map-list" }
  /** Name → fixed-length numbers, e.g. bands → [low, high]. */
  | { kind: "map-tuple"; size: number }
  /** A list of objects, each with its own fields. */
  | { kind: "objects"; schema: JsonSchema }
  | { kind: "json" };

export interface Field {
  name: string;
  title: string;
  description: string;
  required: boolean;
  /** Null allowed (an optional setting that may be "not set"). */
  nullable: boolean;
  default: unknown;
  hasDefault: boolean;
  kind: FieldKind;
  schema: JsonSchema;
}

/** ``schema`` with a top-level ``$ref`` replaced by its definition from ``root``. */
export function deref(schema: JsonSchema, root: JsonSchema): JsonSchema {
  let current = schema;
  for (let i = 0; i < 10 && current.$ref; i++) {
    const name = current.$ref.split("/").pop() ?? "";
    const target = root.$defs?.[name];
    if (!target) break;
    const { $ref: _ignored, ...rest } = current;
    current = { ...target, ...rest };
  }
  return current;
}

function isNull(s: JsonSchema): boolean {
  return s.type === "null";
}

function stripNull(schema: JsonSchema, root: JsonSchema): { schema: JsonSchema; nullable: boolean } {
  const options = schema.anyOf ?? schema.oneOf;
  if (!options) return { schema: deref(schema, root), nullable: false };
  const rest = options.filter((o) => !isNull(o));
  const nullable = rest.length < options.length;
  if (rest.length === 1) {
    const { anyOf: _a, oneOf: _o, ...outer } = schema;
    return { schema: { ...deref(rest[0], root), ...outer }, nullable };
  }
  return { schema: { ...schema, anyOf: rest.map((r) => deref(r, root)) }, nullable };
}

function itemType(s: JsonSchema | undefined): "string" | "number" | null {
  if (!s) return null;
  if (s.type === "string") return "string";
  if (s.type === "number" || s.type === "integer") return "number";
  return null;
}

function tupleShape(s: JsonSchema, root: JsonSchema): { size: number; nullable: boolean[] } | null {
  if (s.type !== "array" || !s.prefixItems) return null;
  const items = s.prefixItems.map((p) => stripNull(p, root));
  if (!items.every((i) => i.schema.type === "number" || i.schema.type === "integer")) return null;
  return { size: items.length, nullable: items.map((i) => i.nullable) };
}

export function fieldKind(schema: JsonSchema, root: JsonSchema): FieldKind {
  const s = schema;
  if (s.anyOf) {
    const enums = s.anyOf.find((o) => o.enum || o.const !== undefined);
    const list = s.anyOf.find((o) => o.type === "array" && itemType(o.items) === "string");
    if (enums && list && s.anyOf.length === 2) {
      return { kind: "enum-or-list", options: (enums.enum ?? [enums.const]).map(String) };
    }
    const numbers = s.anyOf.every((o) => o.type === "number" || o.type === "integer");
    if (numbers) return { kind: "number", integer: false };
    return { kind: "json" };
  }
  if (s.enum) return { kind: "enum", options: s.enum.map(String) };
  if (s.const !== undefined) return { kind: "enum", options: [String(s.const)] };
  switch (s.type) {
    case "string":
      return { kind: "text" };
    case "number":
      return { kind: "number", integer: false };
    case "integer":
      return { kind: "number", integer: true };
    case "boolean":
      return { kind: "boolean" };
    case "array": {
      const tuple = tupleShape(s, root);
      if (tuple) return { kind: "tuple", ...tuple };
      const items = s.items ? deref(s.items, root) : undefined;
      const simple = itemType(items);
      if (simple) return { kind: "list", item: simple };
      if (items?.type === "object" && items.properties) return { kind: "objects", schema: items };
      return { kind: "json" };
    }
    case "object": {
      const extra = typeof s.additionalProperties === "object" ? deref(s.additionalProperties, root) : null;
      if (extra?.type === "array" && itemType(extra.items) === "string") return { kind: "map-list" };
      const tuple = extra ? tupleShape(extra, root) : null;
      if (tuple) return { kind: "map-tuple", size: tuple.size };
      return { kind: "json" };
    }
  }
  return { kind: "json" };
}

/** The fields of an object schema, in its order, without ``hidden`` ones. */
export function fields(schema: JsonSchema, hidden: readonly string[] = []): Field[] {
  const root = schema;
  const required = new Set(schema.required ?? []);
  return Object.entries(schema.properties ?? {})
    .filter(([name]) => !hidden.includes(name))
    .map(([name, raw]) => {
      const { schema: s, nullable } = stripNull(raw, root);
      const hasDefault = "default" in raw;
      return {
        name,
        title: raw.title ?? s.title ?? humanise(name),
        description: raw.description ?? s.description ?? "",
        required: required.has(name),
        nullable,
        default: raw.default,
        hasDefault,
        kind: fieldKind(s, root),
        schema: s,
      };
    });
}

export function humanise(name: string): string {
  const text = name.replace(/_/g, " ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}

// --- values as text ----------------------------------------------------------------------

export function listToText(value: unknown): string {
  return Array.isArray(value) ? value.map((v) => (v === null ? "" : String(v))).join(", ") : "";
}

/** "Fz, Cz , Pz" → ["Fz", "Cz", "Pz"]; numbers when ``numbers``. Null if not valid. */
export function textToList(text: string, numbers = false): (string | number)[] | null {
  const parts = text
    .split(",")
    .map((p) => p.trim())
    .filter(Boolean);
  if (!numbers) return parts;
  const values = parts.map(Number);
  return values.every((v) => Number.isFinite(v)) ? values : null;
}

/** A number typed by the user, or null for empty, or undefined if it is not a number. */
export function parseNumber(text: string, integer: boolean): number | null | undefined {
  const t = text.trim();
  if (t === "") return null;
  const value = Number(t);
  if (!Number.isFinite(value)) return undefined;
  if (integer && !Number.isInteger(value)) return undefined;
  return value;
}

/** What the field shows when the user has not set it. */
export function describeDefault(field: Field): string {
  if (!field.hasDefault) return field.required ? "required" : "";
  const d = field.default;
  if (d === null) return "not set";
  if (Array.isArray(d)) return d.length ? `[${d.map((v) => (v === null ? "start" : String(v))).join(", ")}]` : "none";
  if (typeof d === "object") return JSON.stringify(d);
  if (typeof d === "boolean") return d ? "yes" : "no";
  return String(d);
}

/** ``args`` with ``name`` set to ``value``, or removed when ``value`` is undefined. */
export function withArg(
  args: Record<string, unknown>,
  name: string,
  value: unknown,
): Record<string, unknown> {
  const next = { ...args };
  if (value === undefined) delete next[name];
  else next[name] = value;
  return next;
}
