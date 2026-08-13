import assert from "node:assert/strict";
import { parsePromptsText } from "../custom_nodes/ComfyUI-WanLoopBatch/web/wan_loop_prompt_parser.mjs";

const multiline = Array.from(
  { length: 10 },
  (_, index) => `${index + 1}. subject ${index + 1}\ncamera motion ${index + 1}`,
).join("\n\n");
const parsed = parsePromptsText(multiline);
assert.equal(parsed.length, 10);
assert.equal(parsed[0], "subject 1\ncamera motion 1");
assert.equal(parsed[9], "subject 10\ncamera motion 10");

const japaneseNumbering = multiline.replace("1.", "1：");
assert.equal(parsePromptsText(japaneseNumbering)[0], "subject 1\ncamera motion 1");

const crlf = multiline.replaceAll("\n", "\r\n");
assert.deepEqual(parsePromptsText(crlf), parsed);

const legacy = Array.from({ length: 10 }, (_, index) => `legacy ${index + 1}`).join("\n");
assert.equal(parsePromptsText(legacy)[4], "legacy 5");

const json = JSON.stringify(Array.from({ length: 10 }, (_, index) => `json ${index + 1}`));
assert.equal(parsePromptsText(json)[7], "json 8");

assert.throws(
  () => parsePromptsText(multiline.split("\n\n").slice(0, 9).join("\n\n")),
  /exactly 10 blank-line-separated prompt blocks; found 9/,
);

console.log("prompt parser tests passed");
