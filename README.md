# Paper Notes

论文阅读笔记公开存档，使用 [Quartz v5](https://quartz.jzhao.xyz/) 渲染，部署在 GitHub Pages。

站点地址：https://auyuir.github.io/paper-notes

## 更新笔记

内容从私有 Obsidian vault 同步而来。更新流程：

```bash
bash gpu-learning-journal/sync-paper-notes.sh
cd paper-notes
git add -A && git commit -m "sync: 更新论文笔记" && git push
```

推送后 GitHub Actions 会自动构建并部署。
