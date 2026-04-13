---
name: frontend-uiux-design
description: Design and implement polished frontend UI/UX for this project's medical analytics web interface. Use when Codex needs to create, redesign, restyle, or refine pages, components, dashboards, forms, search flows, results views, or data-visualization screens in the current Thymeleaf plus Bootstrap plus ECharts stack, especially for dashboard, patient search, patient results, and disease prediction experiences.
---

Build frontend work for this repository with a strong UI/UX lens and with awareness of the existing medical analytics product.

## Work Process

1. Identify the page or component being changed and the user task it supports.
2. Read `references/project-ui-context.md` before making design decisions that affect layout, styling, navigation, or visualization.
3. Preserve the current stack unless the user explicitly asks for a framework change:
   - Thymeleaf templates in `src/main/resources/templates/`
   - shared CSS in `src/main/resources/static/css/comon0.css`
   - Bootstrap 5
   - Bootstrap Icons
   - ECharts
   - jQuery for current page behavior
4. Choose a clear aesthetic direction instead of making generic "clean dashboard" edits.
5. Implement real working code, not mockups.
6. Keep desktop and mobile behavior usable.

## Design Rules

- Preserve healthcare-product clarity: prioritize trust, hierarchy, readability, and scanability.
- Avoid generic AI-looking UI choices such as purple gradients, default system-heavy styling, or interchangeable SaaS cards.
- Use one coherent visual direction per task. Keep typography, color, spacing, and motion aligned to that direction.
- Reuse the existing information architecture unless the user asks for workflow changes.
- Favor meaningful motion and hover/focus states over decorative animation spam.
- Treat charts and metric blocks as product surfaces, not decoration.

## Implementation Guidance

- Prefer editing existing templates and shared styles before adding new one-off CSS blocks.
- Keep CSS tokens centralized when introducing new colors, spacing, or shadows.
- Preserve Thymeleaf bindings and server-rendered data hooks.
- Preserve existing IDs, selectors, and chart mount points unless the related JavaScript is updated in the same task.
- When changing chart presentation, verify the container sizing and responsive behavior.
- When forms or search inputs are touched, improve empty, loading, error, and success states when practical.

## Project Notes

- The current visual baseline is Apple-inspired, light, glassy, and data-dashboard oriented.
- The main product surfaces are dashboard, patient search, patient details/results, and disease prediction.
- When redesigning a page, improve visual distinctiveness without breaking the shared navigation and medical-data context.

## Output Expectation

Ship changes that are visually intentional, technically consistent with this repo, and ready to run inside the current Spring Boot application.
