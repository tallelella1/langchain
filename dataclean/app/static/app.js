"use strict";
const $ = id => document.getElementById(id);
const state = {token: null, page: 1, timer: null, generation: 0};
const statusNames = {queued: "排队中", running: "执行中", succeeded: "已完成", failed: "处理失败"};

function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
  $("notice").hidden = false;
}

function logout() {
  clearTimeout(state.timer);
  state.generation++;
  state.token = null;
  $("workspace").hidden = true;
  $("account").hidden = true;
  $("auth").hidden = false;
  $("jobs").replaceChildren();
  $("notice").hidden = true;
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (state.token) headers.set("Authorization", `Bearer ${state.token}`);
  const response = await fetch(path, {...options, headers});
  if (!response.ok) {
    let message = `请求失败 (${response.status})`;
    try {
      const body = await response.json();
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch (_) { /* Keep HTTP error when a response has no JSON body. */ }
    if (response.status === 401 && state.token) logout();
    throw new Error(message);
  }
  return response;
}

async function authenticate(register = false) {
  if (!$("auth-form").reportValidity()) return;
  $("login").disabled = $("register").disabled = true;
  try {
    const username = $("login-name").value;
    const password = $("password").value;
    if (register) await api("/api/auth/register", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({username, password})});
    const response = await api("/api/auth/token", {method: "POST", body: new URLSearchParams({username, password})});
    state.token = (await response.json()).access_token;
    state.page = 1;
    state.generation++;
    $("username").textContent = username.toLowerCase();
    $("password").value = "";
    $("auth").hidden = true;
    $("workspace").hidden = $("account").hidden = false;
    $("notice").hidden = true;
    await loadJobs();
  } catch (error) { notice(error.message, true); }
  finally { $("login").disabled = $("register").disabled = false; }
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

async function download(job, artifact) {
  try {
    const response = await api(`/api/jobs/${job.id}/download/${artifact}`);
    const url = URL.createObjectURL(await response.blob());
    const anchor = element("a");
    anchor.href = url;
    anchor.download = `${job.id}_${{cleaned: "cleaned.csv", report: "report.json", statistics: "statistics.csv", rejected: "rejected.csv"}[artifact]}`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (error) { notice(error.message, true); }
}

function jobRow(job) {
  const tr = element("tr");
  const file = element("td");
  file.append(element("strong", job.filename), element("small", job.id));
  const status = element("td");
  status.append(element("span", statusNames[job.status], `badge ${job.status}`));
  const summary = element("td", undefined, "summary");
  if (job.summary) {
    summary.append(element("div", `保留 ${job.summary.valid_rows} 行 · 异常 ${job.summary.invalid_rows} 行`), element("div", `去重 ${job.summary.duplicate_rows} 行 · ¥ ${job.summary.total_amount}`));
  } else if (job.error) summary.append(element("div", job.error, "error-text"));
  else summary.textContent = "等待处理结果";
  const time = element("td", new Date(job.created_at).toLocaleString("zh-CN", {hour12: false}));
  const actions = element("td");
  const group = element("div", undefined, "row-actions");
  if (job.status === "succeeded") {
    for (const [artifact, label] of Object.entries({cleaned: "清洗数据", report: "JSON 报表", statistics: "统计 CSV", rejected: "异常明细"})) {
      const button = element("button", label);
      button.addEventListener("click", () => download(job, artifact));
      group.append(button);
    }
  } else if (job.status === "failed") {
    const retry = element("button", "重新处理");
    retry.addEventListener("click", async () => {
      retry.disabled = true;
      try { await api(`/api/jobs/${job.id}/retry`, {method: "POST"}); await loadJobs(); }
      catch (error) { notice(error.message, true); retry.disabled = false; }
    });
    group.append(retry);
  } else group.textContent = "完成后可下载";
  actions.append(group);
  tr.append(file, status, summary, time, actions);
  return tr;
}

async function loadJobs() {
  clearTimeout(state.timer);
  if (!state.token) return;
  const generation = ++state.generation;
  try {
    const query = new URLSearchParams({page: state.page, page_size: 10});
    if ($("status-filter").value) query.set("status", $("status-filter").value);
    const result = await (await api(`/api/jobs?${query}`)).json();
    if (generation !== state.generation || !state.token) return;
    $("jobs").replaceChildren(...result.items.map(jobRow));
    $("count").textContent = result.total;
    $("empty").hidden = result.items.length > 0;
    $("empty").textContent = $("status-filter").value ? "暂无该状态的任务。" : "还没有处理记录，上传一份 CSV 开始。";
    $("page-info").textContent = `第 ${state.page} / ${Math.max(1, Math.ceil(result.total / 10))} 页，共 ${result.total} 条`;
    $("prev").disabled = state.page <= 1;
    $("next").disabled = state.page * 10 >= result.total;
    if (result.items.some(job => ["queued", "running"].includes(job.status))) state.timer = setTimeout(loadJobs, 2000);
  } catch (error) { if (generation === state.generation) notice(error.message, true); }
}

$("auth-form").addEventListener("submit", event => { event.preventDefault(); authenticate(); });
$("register").addEventListener("click", () => authenticate(true));
$("logout").addEventListener("click", logout);
$("csv-file").addEventListener("change", () => { $("file-name").textContent = $("csv-file").files[0]?.name || "点击选择文件"; });
$("upload-form").addEventListener("submit", async event => {
  event.preventDefault();
  $("upload").disabled = true;
  try {
    const body = new FormData();
    body.append("file", $("csv-file").files[0]);
    const response = await api("/api/jobs", {method: "POST", body});
    const job = await response.json();
    notice(response.headers.get("Idempotent-Replayed") ? `已找到相同文件的任务：${job.id}` : `任务已提交：${job.id}`);
    state.page = 1;
    $("status-filter").value = "";
    await loadJobs();
  } catch (error) { notice(error.message, true); }
  finally { $("upload").disabled = false; }
});
$("refresh").addEventListener("click", loadJobs);
$("status-filter").addEventListener("change", () => { state.page = 1; loadJobs(); });
$("prev").addEventListener("click", () => { state.page--; loadJobs(); });
$("next").addEventListener("click", () => { state.page++; loadJobs(); });
