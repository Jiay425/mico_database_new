# Design System Inspired by Stripe

## 1. Visual Theme & Atmosphere
Stripe's aesthetic is the gold standard for premium developer tools and financial software. It balances stark, pristine white canvases with subtle, slightly warm slate grays. Unlike Apple's immersive darkness or heavy glassmorphism, Stripe uses solid white cards (`#ffffff`) placed on slightly off-white backgrounds (`#f6f9fc`) to create a clean, clinical, yet approachable data environment. 

The primary accent is "Blurkle" (`#635BFF`), an energetic blue-purple that feels distinctly modern. Shadows do the heavy lifting for hierarchy—Stripe is legendary for its deep, incredibly soft, multi-layered colorful shadows that lift interactive elements effortlessly off the page.

### Key Characteristics:
- Inter typography with distinct weight distributions (400 for body, 500/600 for important UI elements).
- Pure white cards on `#f6f9fc` (Slate) background.
- Crisp, near-invisible borders (`#e3e8ee`) combined with high-quality box shadows.
- Distinctly structured "Bento" layouts separated by whitespace rather than heavy lines.
- Slanted backgrounds and mesh gradients used sparingly as subtle accents.
- Bright blue/purple accents (`#635BFF`) communicating interaction.

## 2. Color Palette & Roles

### Base & Backgrounds
- **Primary Background**: `#f6f9fc` (Slate 50) - The cool, clinical off-white for the main app shell.
- **Card/Surface**: `#ffffff` (White) - Pristine white for all data panels and active components.
- **Secondary Surface**: `#f3f4f6` - Hover states for list items or secondary cards.

### Text & Typography
- **Primary Text**: `#0a2540` (Deep Slate) - Almost black, incredibly crisp for primary headings and prominent data.
- **Secondary Text**: `#425466` (Slate) - For sub-navigation, descriptions, and secondary metrics.
- **Tertiary Text**: `#8898aa` (Cool Gray) - For muted labels or placeholders.

### Interactive & Brand
- **Blurkle (Brand & CTA)**: `#635BFF` - Primary brand color, used for main buttons and active states.
- **Blurkle Hover**: `#0A2540` (Buttons often invert to dark slate, or brighten the blurkle).
- **Secondary Action**: `#0a2540` background on white, or translucent white on dark.

### Sub-Brands (Status / Notifications)
- **Positive / Success**: `#00d4ff` or `#00c0f9` (Light vibrant cyan/blue) or Stripe Green `#24b47e`.
- **Warning**: `#ffb845` or Stripe Orange.
- **Danger / Alert**: `#e25950` (Crisp red).

## 3. Shadows & Elevation
Stripe uses a meticulously crafted multiple-shadow layer system.

- **Card Level 1 (Static panels, Bentos)**
  `box-shadow: 0 2px 4px rgba(0,0,0,0.02), 0 1px 0 rgba(0,0,0,0.03);`
  (Very subtle, mostly relies on the `#e3e8ee` border)
- **Card Level 2 (Hover or Hero Elements)**
  `box-shadow: 0 13px 27px -5px rgba(50,50,93,0.25), 0 8px 16px -8px rgba(0,0,0,0.3);`
- **Dropdowns & Modals**
  `box-shadow: 0 50px 100px -20px rgba(50,50,93,0.25), 0 30px 60px -30px rgba(0,0,0,0.3);`

## 4. Component Stylings

### Buttons
**Primary (Blurkle)**
- Background: `#635BFF`
- Text: `#FFFFFF`
- Padding: 8px 16px
- Radius: 100px (Pill) or 4px/8px depending on context. For dashboards, usually 8px.
- Shadow: `0 4px 6px rgba(50,50,93,.11), 0 1px 3px rgba(0,0,0,.08);`
- Hover: translateY(-1px), shadow intensifies.

**Secondary (Slate View)**
- Background: `#ffffff`
- Color: `#0a2540`
- Border: none or strict 1px solid `#e3e8ee`
- Shadow: `0 2px 5px 0 rgba(50,50,93,.1), 0 1px 1px 0 rgba(0,0,0,.07)`

### Cards & Panels
- Background: `#ffffff`
- Border: `1px solid #e3e8ee`
- Radius: `12px` for main panels, `16px` or `24px` for hero.
- Inner Spacing: Generous `24px`.

## 5. UI Application to Mico Dashboard
- Replace the dark/glassmorphic hero header with a pristine white or slate container.
- Drop all `.ios-primary` and iOS blur systems.
- ECharts styling should use clean lines, matching the Slack/Stripe data palettes (slate grids, blurkle bars, etc.).
- Convert bento grids into isolated white cards sitting on `#f6f9fc` background.
