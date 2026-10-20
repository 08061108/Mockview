# Mockview Web

当前受支持客户端：React 19、TypeScript、Vite 6，Node 22.x。

```bash
npm ci
npm run dev
npm run build
```

开发环境见 `.env.example`；Docker 构建用 `/api`，Nginx 同源代理。VITE 环境变量是构建时公开值，不放密钥。WS 使用首帧 auth，不把 JWT 放 URL。

完整启动与安全说明以 [部署指南](../DEPLOYMENT.md) 为准，演示与限制见 [演示路线](../docs/demo.md) 和 [限制](../docs/limitations.md)。旧 frontend/Gradio 不是受支持入口。
