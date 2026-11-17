import { describe, expect, it } from "vitest";
import { cleanLabel, findMapping, guessFromName, parseMappingCsv, stem } from "./filenames";

describe("guessFromName", () => {
  it.each([
    ["D:\\new_study\\s41_sess1.bdf", { subject: "41", session: "1", task: "", run: "" }],
    ["/data/sub-007_ses-02_task-P3_run-1_eeg.edf", { subject: "007", session: "02", task: "P3", run: "1" }],
    ["P012_visit2.vhdr", { subject: "012", session: "2", task: "", run: "" }],
    ["017.edf", { subject: "017", session: "", task: "", run: "" }],
    ["participant_23_oddball.set", { subject: "23", session: "", task: "", run: "" }],
  ])("%s", (path, expected) => {
    expect(guessFromName(path)).toEqual(expected);
  });

  it("leaves names without a pattern empty", () => {
    expect(guessFromName("pilot_final.bdf").subject).toBe("");
    expect(guessFromName("redo.bdf").subject).toBe("");
  });
});

describe("stem", () => {
  it("drops folders and the format's extension", () => {
    expect(stem("C:\\a\\b\\rec.fif.gz")).toBe("rec");
    expect(stem("/x/y/rec.mff/")).toBe("rec");
  });
});

describe("cleanLabel", () => {
  it("keeps letters and digits", () => {
    expect(cleanLabel("P3 odd-ball_2")).toBe("P3oddball2");
  });
});

describe("parseMappingCsv", () => {
  it("reads the columns, quoted cells and a byte-order mark", () => {
    const text = '\uFEFFsource,subject_id,task,session\r\n"C:\\data\\a, b.bdf",P01,P3,1\nb.bdf,P02,P3,\n\n';
    expect(parseMappingCsv(text)).toEqual([
      { source: "C:\\data\\a, b.bdf", subject_id: "P01", task: "P3", session: "1", run: "", line_freq: "" },
      { source: "b.bdf", subject_id: "P02", task: "P3", session: "", run: "", line_freq: "" },
    ]);
  });

  it("names missing columns", () => {
    expect(() => parseMappingCsv("source,task\na,b\n")).toThrow("subject_id");
  });

  it("matches by path, then by file name", () => {
    const rows = parseMappingCsv("source,subject_id,task\nb.bdf,P02,P3\nD:/x/a.bdf,P01,P3\n");
    expect(findMapping(rows, "D:\\x\\a.bdf")?.subject_id).toBe("P01");
    expect(findMapping(rows, "E:\\other\\B.BDF")?.subject_id).toBe("P02");
    expect(findMapping(rows, "c.bdf")).toBeUndefined();
  });
});
