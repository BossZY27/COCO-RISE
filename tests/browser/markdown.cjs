// Run with node tests/browser/markdown.cjs (no packages required).
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('app/static/chat.html', 'utf8');
const source = html.slice(html.indexOf('function renderAnswer('), html.indexOf('function renderTurn('));
class Element {
  constructor(name) { this.localName = name; this.children = []; }
  appendChild(child) { this.children.push(child); }
  set textContent(value) { this.children = [value]; }
  get textContent() { return this.children.map(child => typeof child === 'string' ? child : child.textContent).join(''); }
}
const document = {
  createElement: name => new Element(name),
  createTextNode: value => value,
};
const context = vm.createContext({document});
vm.runInContext(source, context);
const target = new Element('div');
context.renderAnswer(target, '### ทุนวิจัย\n\n- **ทุน ก** (ข้อ 1.1)\n- ทุน ข\n\n3. ขั้นตอน\n\nข้อความ *เน้น* และ `โค้ด`\n\n```html\n<img src=x onerror=alert(1)>\n```\n\n<script>alert(1)</script> [คลิก](javascript:alert(1))');
assert.equal(target.children[0].localName, 'h3');
assert.equal(target.children[1].localName, 'ul');
assert.equal(target.children[1].children.length, 2);
assert.equal(target.children[1].children[0].children[1].localName, 'strong');
assert.equal(target.children[2].localName, 'ol');
assert.equal(target.children[2].start, 3);
assert(target.textContent.includes('ทุน ก (ข้อ 1.1)'));
assert(target.textContent.includes('<script>alert(1)</script>'));
assert(target.textContent.includes('<img src=x onerror=alert(1)>'));
function inspect(element) {
  assert(!['script', 'img', 'a', 'iframe'].includes(element.localName));
  for (const child of element.children) if (typeof child !== 'string') inspect(child);
}
inspect(target);
for (const script of html.matchAll(/<script>([\s\S]*?)<\/script>/g)) new vm.Script(script[1]);
console.log('Markdown formatting, inert HTML/URLs, and chat syntax passed');
