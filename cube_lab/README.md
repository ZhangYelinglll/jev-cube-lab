# Jev 魔方实验室

Jev 根据六面状态逐步选择 18 个面转动作之一。网页展示真实返回概率、置信度、旋转动画、操作轨迹和按打乱步数分组的实验记录。没有求解器或隐式接管；允许连续同面转动（如 U → U）；每步排除立即逆操作，并用状态模拟预演，排除返回本轮已出现状态的动作；若所有候选都会重复，则仅保留立即逆操作限制，继续运行，其余动作由 Jev 自主选择；模型看不到打乱步骤。每次请求都会附带本轮起始状态、此前每一步的动作和执行后状态，作为决策背景。

## 运行

在项目根目录的 `.env` 中设置 `TYPESAFE_API_KEY`，可选设置 `TYPESAFE_MODEL`（默认 `jev-1.13.0`）。

```bash
npm --prefix cube_lab ci
npm --prefix cube_lab run build
uv run --env-file .env python cube_server.py
```

打开 http://127.0.0.1:8765 。如果项目运行在远程服务器，请把 8765 端口转发到本机。服务默认只监听本机；密钥仅由 Python 服务读取，静态资源仅开放应用页面、样式及构建产物。

## 使用

选择打乱步数，点击“执行一步”或“开始自动实验”。暂停会等待当前请求及动画结束，避免产生半步状态。还原成功或达到步数上限会结束本轮；重复状态不会停止。暂时连接失败会自动重试两次（每步最多三次请求），期间不改变魔方；仍失败则暂停，可手动继续。暂停也会取消尚未开始的重试。

点击操作轨迹回看对应状态和概率；重放不消耗 API。修改难度或上限会新建一轮。记录保存在当前浏览器，保留最近 100 轮，可导出 JSON。未完成、提前结束、网络错误不计入成功率。打乱步数不等于最短解距离；小样本结果不能视为整体能力评估。

状态使用 `cubejs/lib/cube.js`（不导入 solve 模块），动画使用固定版本 cubing.js。两个模块使用相同的 Singmaster 操作序列。返回值由服务端校验后才执行。置信度衡量概率分布集中程度，不是还原正确率。页面展示可观测输入、输出和结果，不生成推理解释。

## 检查

```bash
uv run python check_cube_lab.py
npm --prefix cube_lab run check
# 启动服务后；浏览器检查使用明确标注的离线接口替身，不调用 Jev。
cd cube_lab
npx playwright install chromium
node check_browser.mjs
# 或设置 BROWSER_BIN 指向已有 Chromium 可执行文件。
```

`node evaluate.mjs` 会真实调用运行中的 Jev 接口，测试 U、R、F'、R U、R U F 五个状态，每轮最多 10 步，将原始响应与轨迹保存到 `data/jev_cube_smoke.json`（需在 cube_lab 目录运行）。该命令消耗 API 用量，不在普通测试中自动运行。
