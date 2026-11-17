/** A form generated from a tool's (or a template's) JSON schema. It edits only the settings
 * the user sets; the rest keep their defaults, shown greyed in each field. */

import { useEffect, useState } from "react";
import type { JsonSchema } from "../engine/types";
import {
  type Field,
  describeDefault,
  fields as fieldsOf,
  listToText,
  parseNumber,
  textToList,
  withArg,
} from "../lib/schema";
import * as Icon from "./icons";

export function SchemaForm({
  schema,
  value,
  onChange,
  hidden = [],
  problems = {},
  readOnly = false,
  suggestions = {},
}: {
  schema: JsonSchema;
  value: Record<string, unknown>;
  onChange: (value: Record<string, unknown>) => void;
  hidden?: readonly string[];
  /** Problems by field name, shown under the field. */
  problems?: Record<string, string>;
  readOnly?: boolean;
  /** Words to offer while typing, by kind: "channels", "events", "conditions". */
  suggestions?: Record<string, string[]>;
}) {
  const list = fieldsOf(schema, hidden);
  if (list.length === 0) return <p className="small muted">This step has no settings.</p>;
  return (
    <div className="form-grid">
      {list.map((field) => (
        <FieldRow
          key={field.name}
          field={field}
          root={schema}
          value={value[field.name]}
          set={field.name in value}
          problem={problems[field.name]}
          readOnly={readOnly}
          suggestions={suggestions}
          onChange={(v) => onChange(withArg(value, field.name, v))}
        />
      ))}
    </div>
  );
}

interface FieldProps {
  field: Field;
  root: JsonSchema;
  value: unknown;
  set: boolean;
  problem?: string;
  readOnly: boolean;
  suggestions: Record<string, string[]>;
  /** undefined: back to the default. */
  onChange: (value: unknown) => void;
}

function FieldRow(props: FieldProps) {
  const { field, set, problem, readOnly, onChange } = props;
  const wide = ["map-list", "map-tuple", "objects", "json"].includes(field.kind.kind);
  const id = `f-${field.name}`;
  const missing = field.required && !set;
  return (
    <div className={`form-field${wide ? " wide" : ""}`}>
      <div className="row" style={{ gap: 6 }}>
        <label htmlFor={id} className="form-label">
          {field.title}
          {field.required ? <span title="Required" style={{ color: "var(--warn-text)" }}> *</span> : null}
        </label>
        <span className="spacer" />
        {set && !readOnly && field.hasDefault ? (
          <button type="button" className="btn link small" onClick={() => onChange(undefined)} title="Use the default">
            Default
          </button>
        ) : null}
      </div>
      <Control {...props} id={id} />
      {field.description ? <div className="tiny muted">{field.description}</div> : null}
      {problem ? (
        <div className="tiny" style={{ color: "var(--danger)" }}>
          {problem}
        </div>
      ) : missing ? (
        <div className="tiny" style={{ color: "var(--warn-text)" }}>
          Required
        </div>
      ) : null}
    </div>
  );
}

function Control(props: FieldProps & { id: string }) {
  const { field, value, set, readOnly, onChange, id, suggestions } = props;
  const placeholder = describeDefault(field);
  const kind = field.kind;
  const listId = suggestionList(field, suggestions);

  switch (kind.kind) {
    case "text":
      return (
        <>
          <input
            id={id}
            className="text"
            disabled={readOnly}
            value={set && value != null ? String(value) : ""}
            placeholder={placeholder}
            list={listId?.id}
            onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value)}
          />
          {listId?.element}
        </>
      );
    case "number":
      return (
        <NumberInput
          id={id}
          integer={kind.integer}
          value={set ? (value as number | null) : undefined}
          placeholder={placeholder}
          readOnly={readOnly}
          onChange={onChange}
        />
      );
    case "boolean":
    case "enum": {
      const options = kind.kind === "boolean" ? ["true", "false"] : kind.options;
      const label = (o: string) => (o === "true" ? "Yes" : o === "false" ? "No" : o);
      return (
        <select
          id={id}
          className="text"
          disabled={readOnly}
          value={set ? String(value) : ""}
          onChange={(e) => {
            const v = e.target.value;
            onChange(v === "" ? undefined : kind.kind === "boolean" ? v === "true" : v);
          }}
        >
          <option value="">{field.hasDefault ? `Default (${placeholder})` : "Choose…"}</option>
          {options.map((o) => (
            <option key={o} value={o}>
              {label(o)}
            </option>
          ))}
        </select>
      );
    }
    case "enum-or-list": {
      const isList = Array.isArray(value);
      return (
        <div className="stack" style={{ gap: 6 }}>
          <select
            id={id}
            className="text"
            disabled={readOnly}
            value={!set ? "" : isList ? "__list" : String(value)}
            onChange={(e) => {
              const v = e.target.value;
              onChange(v === "" ? undefined : v === "__list" ? [] : v);
            }}
          >
            <option value="">{field.hasDefault ? `Default (${placeholder})` : "Choose…"}</option>
            {kind.options.map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
            <option value="__list">Channels…</option>
          </select>
          {isList ? (
            <ListInput
              value={value as unknown[]}
              numbers={false}
              placeholder="e.g. P9, P10"
              readOnly={readOnly}
              onChange={onChange}
            />
          ) : null}
        </div>
      );
    }
    case "list":
      return (
        <ListInput
          id={id}
          value={set ? (value as unknown[]) : undefined}
          numbers={kind.item === "number"}
          placeholder={placeholder || "comma-separated"}
          readOnly={readOnly}
          onChange={onChange}
        />
      );
    case "tuple":
      return (
        <TupleInput
          size={kind.size}
          nullable={kind.nullable}
          wholeNullable={field.nullable}
          value={set ? (value as (number | null)[] | null) : undefined}
          fallback={field.default as (number | null)[] | null | undefined}
          readOnly={readOnly}
          onChange={onChange}
        />
      );
    case "map-list":
    case "map-tuple":
      return (
        <MapInput
          value={set ? (value as Record<string, unknown>) : undefined}
          size={kind.kind === "map-tuple" ? kind.size : null}
          readOnly={readOnly}
          onChange={onChange}
          eventSuggestions={suggestions.events}
        />
      );
    case "objects":
      return (
        <ObjectsInput
          schema={kind.schema}
          root={props.root}
          value={set ? (value as Record<string, unknown>[]) : undefined}
          readOnly={readOnly}
          onChange={onChange}
        />
      );
    case "json":
      return <JsonInput value={set ? value : undefined} placeholder={placeholder} readOnly={readOnly} onChange={onChange} />;
  }
}

function suggestionList(field: Field, suggestions: Record<string, string[]>) {
  const name = field.name.toLowerCase();
  const words =
    name.includes("channel") ? suggestions.channels : name.startsWith("condition") ? suggestions.conditions : undefined;
  if (!words?.length) return null;
  const id = `dl-${field.name}`;
  return {
    id,
    element: (
      <datalist id={id}>
        {words.map((w) => (
          <option key={w} value={w} />
        ))}
      </datalist>
    ),
  };
}

/** A text box for a number, keeping what is typed (e.g. "0.") until it is a number. */
function NumberInput({
  id,
  integer,
  value,
  placeholder,
  readOnly,
  onChange,
}: {
  id?: string;
  integer: boolean;
  value: number | null | undefined;
  placeholder: string;
  readOnly: boolean;
  onChange: (value: unknown) => void;
}) {
  const shown = value == null ? "" : String(value);
  const [text, setText] = useState(shown);
  const [bad, setBad] = useState(false);
  useEffect(() => {
    if (parseNumber(text, integer) !== (value ?? null)) setText(shown);
  }, [shown]);
  return (
    <input
      id={id}
      className={`text mono${bad ? " invalid" : ""}`}
      inputMode="decimal"
      disabled={readOnly}
      value={text}
      placeholder={placeholder}
      onChange={(e) => {
        setText(e.target.value);
        const parsed = parseNumber(e.target.value, integer);
        setBad(parsed === undefined);
        if (parsed !== undefined) onChange(parsed === null ? undefined : parsed);
      }}
    />
  );
}

function ListInput({
  id,
  value,
  numbers,
  placeholder,
  readOnly,
  onChange,
}: {
  id?: string;
  value: unknown[] | undefined;
  numbers: boolean;
  placeholder: string;
  readOnly: boolean;
  onChange: (value: unknown) => void;
}) {
  const shown = value === undefined ? "" : listToText(value);
  const [text, setText] = useState(shown);
  const [bad, setBad] = useState(false);
  useEffect(() => {
    if (listToText(textToList(text, numbers) ?? []) !== shown) setText(shown);
  }, [shown]);
  return (
    <input
      id={id}
      className={`text mono${bad ? " invalid" : ""}`}
      disabled={readOnly}
      value={text}
      placeholder={placeholder}
      onChange={(e) => {
        setText(e.target.value);
        if (e.target.value.trim() === "") {
          setBad(false);
          onChange(undefined);
          return;
        }
        const parsed = textToList(e.target.value, numbers);
        setBad(parsed === null);
        if (parsed) onChange(parsed);
      }}
    />
  );
}

function TupleInput({
  size,
  nullable,
  wholeNullable,
  value,
  fallback,
  readOnly,
  onChange,
}: {
  size: number;
  nullable: boolean[];
  wholeNullable: boolean;
  value: (number | null)[] | null | undefined;
  fallback: (number | null)[] | null | undefined;
  readOnly: boolean;
  onChange: (value: unknown) => void;
}) {
  const none = value === null;
  const current = Array.isArray(value) ? value : Array.isArray(fallback) ? fallback : Array(size).fill(null);
  const labels = size === 2 ? ["from", "to"] : current.map((_, i) => `${i + 1}`);
  return (
    <div className="row" style={{ gap: 8, flexWrap: "nowrap" }}>
      {none
        ? null
        : Array.from({ length: size }, (_, i) => (
            <label key={i} className="row tiny muted" style={{ gap: 4, flexWrap: "nowrap" }}>
              {labels[i]}
              <NumberInput
                integer={false}
                value={value === undefined ? undefined : current[i]}
                placeholder={nullable[i] && current[i] === null ? "start/end" : String(current[i] ?? "")}
                readOnly={readOnly}
                onChange={(v) => {
                  const next = [...current];
                  next[i] = v === undefined ? (nullable[i] ? null : current[i]) : (v as number);
                  onChange(next);
                }}
              />
            </label>
          ))}
      {wholeNullable ? (
        <label className="row small" style={{ gap: 6, flexWrap: "nowrap" }}>
          <input
            type="checkbox"
            disabled={readOnly}
            checked={none}
            onChange={(e) => onChange(e.target.checked ? null : undefined)}
          />
          None
        </label>
      ) : null}
    </div>
  );
}

function MapInput({
  value,
  size,
  readOnly,
  onChange,
  eventSuggestions,
}: {
  value: Record<string, unknown> | undefined;
  size: number | null;
  readOnly: boolean;
  onChange: (value: unknown) => void;
  eventSuggestions?: string[];
}) {
  const rows = Object.entries(value ?? {});
  const update = (next: [string, unknown][]) => onChange(next.length ? Object.fromEntries(next) : undefined);
  return (
    <div className="stack" style={{ gap: 6 }}>
      {rows.map(([key, item], i) => (
        <div key={i} className="row" style={{ gap: 6, flexWrap: "nowrap" }}>
          <input
            className="text mono"
            style={{ width: 140 }}
            aria-label="Name"
            disabled={readOnly}
            value={key}
            placeholder="name"
            onChange={(e) => update(rows.map((r, j) => (j === i ? [e.target.value, r[1]] : r)))}
          />
          <span className="muted">→</span>
          <div style={{ flex: 1 }}>
            {size === null ? (
              <ListInput
                value={item as unknown[]}
                numbers={false}
                placeholder={eventSuggestions?.length ? `e.g. ${eventSuggestions.slice(0, 2).join(", ")}` : "events, comma-separated"}
                readOnly={readOnly}
                onChange={(v) => update(rows.map((r, j) => (j === i ? [r[0], v ?? []] : r)))}
              />
            ) : (
              <TupleInput
                size={size}
                nullable={Array(size).fill(false)}
                wholeNullable={false}
                value={item as number[]}
                fallback={undefined}
                readOnly={readOnly}
                onChange={(v) => update(rows.map((r, j) => (j === i ? [r[0], v] : r)))}
              />
            )}
          </div>
          {readOnly ? null : (
            <button type="button" className="icon-btn" aria-label="Remove" onClick={() => update(rows.filter((_, j) => j !== i))}>
              <Icon.Close />
            </button>
          )}
        </div>
      ))}
      {readOnly ? null : (
        <div>
          <button
            type="button"
            className="btn small"
            onClick={() => update([...rows, [`item${rows.length + 1}`, size === null ? [] : Array(size).fill(0)]])}
          >
            <Icon.Plus /> Add
          </button>
        </div>
      )}
      {eventSuggestions?.length && size === null ? (
        <div className="tiny muted">Events in the data: {eventSuggestions.slice(0, 12).join(", ")}</div>
      ) : null}
    </div>
  );
}

function ObjectsInput({
  schema,
  root,
  value,
  readOnly,
  onChange,
}: {
  schema: JsonSchema;
  root: JsonSchema;
  value: Record<string, unknown>[] | undefined;
  readOnly: boolean;
  onChange: (value: unknown) => void;
}) {
  const items = value ?? [];
  const withDefs = { ...schema, $defs: root.$defs };
  return (
    <div className="stack" style={{ gap: 8 }}>
      {items.map((item, i) => (
        <div key={i} className="card compact" style={{ background: "var(--surface-2)" }}>
          <div className="row">
            <span className="small muted">{i + 1}</span>
            <span className="spacer" />
            {readOnly ? null : (
              <button
                type="button"
                className="icon-btn"
                aria-label="Remove"
                onClick={() => onChange(items.length > 1 ? items.filter((_, j) => j !== i) : undefined)}
              >
                <Icon.Close />
              </button>
            )}
          </div>
          <SchemaForm
            schema={withDefs}
            value={item}
            readOnly={readOnly}
            onChange={(v) => onChange(items.map((x, j) => (j === i ? v : x)))}
          />
        </div>
      ))}
      {readOnly ? null : (
        <div>
          <button type="button" className="btn small" onClick={() => onChange([...items, {}])}>
            <Icon.Plus /> Add
          </button>
        </div>
      )}
    </div>
  );
}

function JsonInput({
  value,
  placeholder,
  readOnly,
  onChange,
}: {
  value: unknown;
  placeholder: string;
  readOnly: boolean;
  onChange: (value: unknown) => void;
}) {
  const shown = value === undefined ? "" : JSON.stringify(value);
  const [text, setText] = useState(shown);
  const [bad, setBad] = useState(false);
  return (
    <textarea
      className={`text mono${bad ? " invalid" : ""}`}
      rows={3}
      disabled={readOnly}
      value={text}
      placeholder={placeholder || "JSON"}
      onChange={(e) => {
        setText(e.target.value);
        if (!e.target.value.trim()) {
          setBad(false);
          onChange(undefined);
          return;
        }
        try {
          onChange(JSON.parse(e.target.value));
          setBad(false);
        } catch {
          setBad(true);
        }
      }}
    />
  );
}
