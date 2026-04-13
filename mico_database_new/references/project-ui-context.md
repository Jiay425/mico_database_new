# Project UI Context

Read this file when working on the current repository's frontend UX, layout, or styling.

## Product Shape

- Domain: medical and microbiome analytics
- Users: researchers, clinicians, operators, or internal analysts reviewing patient and disease data
- Product tone: trustworthy, precise, modern, and data-heavy
- Current visual baseline: Apple-inspired light UI with glassmorphism, rounded cards, soft shadows, and analytics widgets

## Current Tech Stack

- Server rendering: Thymeleaf
- Styling: Bootstrap 5 plus shared custom CSS in `src/main/resources/static/css/comon0.css`
- Icons: Bootstrap Icons
- Charts: ECharts
- Page behavior: mostly jQuery and inline page scripts

## Main Frontend Surfaces

### Dashboard

- File: `src/main/resources/templates/index.html`
- Purpose: high-level hospital or analytics overview
- Common elements: KPI cards, charts, time widget, navbar, summary modules

### Search Entry

- File: `src/main/resources/templates/search_input.html`
- Purpose: quick patient lookup by ID, name, or disease
- UX character: centered hero search with fast access affordances

### Patient Results

- File: `src/main/resources/templates/patient_results.html`
- Purpose: either list matching patients or show one patient's profile and analysis
- UX character: mixed table plus detail dashboard with charts and lab-like data tables

### Disease Prediction

- File: `src/main/resources/templates/disease_prediction.html`
- Purpose: input microbiome data and render prediction results plus visual analysis
- UX character: tool-like workspace with input panel, status badge, and results visualizations

## Shared Style Baseline

- Shared CSS file: `src/main/resources/static/css/comon0.css`
- Existing tokens include:
  - `--ios-bg`
  - `--ios-surface`
  - `--ios-primary`
  - `--ios-success`
  - `--ios-text`
  - `--ios-text-sec`
  - `--ios-border`
- Existing patterns include:
  - glass navbar
  - rounded card surfaces
  - soft shadows
  - pill navigation and buttons
  - restrained motion

## Design Constraints

- Do not break Thymeleaf attributes or server-side data injection.
- Do not rename IDs used by existing JavaScript unless the JS is updated too.
- Keep data-heavy pages readable on laptop widths.
- Keep mobile layouts functional even when the desktop layout is more ambitious.
- Prefer strengthening hierarchy and polish over adding many new components.

## Good Directions For This Repo

- premium clinical analytics
- editorial data storytelling
- refined control-room dashboards
- focused search-and-inspect workflows

## Directions To Avoid

- cartoonish health-tech styling
- generic AI purple gradients
- overly dark hacker dashboards unless explicitly requested
- visual ideas that reduce chart legibility or table readability
