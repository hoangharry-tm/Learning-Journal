<a id="top"></a>

# 👨🏻‍🎨 HTML/CSS

_Table of Contents:_

- [✍🏻 Topic 1: CSS Text](#-topic-1-css-text)
  - [Text Styling](#text-styling)
  - [Text Spacing](#text-spacing)
  - [Text Size](#text-size)
  - [Font Family](#font-family)
  - [Text Color](#text-color)
  - [Documentation Example](#documentation-example)
- [🪴 Topic 2: CSS Selectors](#-topic-2-css-selectors)
  - [The Type Selectors & The Cascade](#the-type-selectors--the-cascade)
  - [Grouping Selectors](#grouping-selectors)
  - [ID & Class Selectors](#id--class-selectors)
  - [Pseudo-class Selectors](#pseudo-class-selectors)
  - [Combinator Selectors](#combinator-selectors)
  - [Specificity](#specificity)
  - [Inheritance](#inheritance)
  - [Pseudo-elements Selectors](#pseudo-elements-selectors)
- [🌷 Topic 3: CSS Box Model](#-topic-3-css-box-model)
  - [Background Color](#background-color)
  - [Width & Height](#width--height)
  - [Border](#border)
  - [Box Sizing](#box-sizing)
  - [Margin](#margin)
  - [Display](#display)
  - [CSS Reset](#css-reset)
  - [Main Container](#main-container)
  - [Width](#width)
  - [Height](#height)
  - [Maximum & Minimum Properties](#maximum--minimum-properties)
- [👨🏻‍🎨 Topic 4: CSS Units](#-topic-4-css-units)
  - [Absolute & Relative Units](#absolute--relative-units)
  - [Percentages](#percentages)
  - [Rems & Ems](#rems--ems)
  - [VH & VW](#vh--vw)
- [🧸 Topic 5: CSS Functions](#-topic-5-css-functions)
  - [CSS Variables](#css-variables)
  - [CSS Calculations](#css-calculations)
- [🤯 Topic 6: Flexbox](#-topic-6-flexbox)
  - [Introduction to Flexbox](#introduction-to-flexbox)
  - [Normal Flow vs. Flexbox](#normal-flow-vs-flexbox)
  - [Justify Content](#justify-content)
  - [Align Items](#align-items)
  - [Gap](#gap)
  - [Nested Flexbox](#nested-flexbox)
  - [Centering](#centering)
  - [Flex Children](#flex-children)
  - [Flex Wrap & Align Content](#flex-wrap--align-content)
- [📺 Topic 7: CSS Grid](#-topic-7-css-grid)
  - [Introduction to CSS Grid](#introduction-to-css-grid)
  - [Grid Columns & Rows](#grid-columns--rows)
  - [Grid Gap](#grid-gap)
  - [Grid _Cell_ Alignment](#grid-cell-alignment)
  - [Grid _Container_ Alignment](#grid-container-alignment)
  - [Grid Items](#grid-items)
- [👟 Topic 8: Responsive Design](#-topic-8-responsive-design)
  - [Media Queries](#media-queries)
  - [Relative Units](#relative-units)
  - [Breakpoints](#breakpoints)
- [🍓 Topic 9: Positioning](#-topic-9-positioning)
  - [`position` Property](#position-property)
  - [Static Positioning](#static-positioning)
  - [Absolute Positioning](#absolute-positioning)
  - [Fixed & Sticky Positioning](#fixed--sticky-positioning)
  - [Positioning Guidance](#positioning-guidance)
  - [Z-index & Stacking Context](#z-index--stacking-context)
  - [Transform](#transform)
- [✍🏻 Topic 10: Shadows & Transitions](#-topic-10-shadows--transitions)
  - [Shadow Property](#shadow-property)
  - [Shadow Guidance](#shadow-guidance)
  - [Flat Design vs. Shadows](#flat-design-vs-shadows)
  - [Transitions](#transitions)

## ✍🏻 Topic 1: CSS Text

In this topic we will learn about some text properties such as styling, spacing,
family, color, and size.

### Text Styling

- _Font Weight:_ it sets the thickness of the character. In CSS, `font-weight` is
  a property and `bold` is a value (i.e. `font-weight: bold`). Although you can
  use name values like `bold` or `normal`, it is better to use numerical values
  which range from `100 - 900`. Note that for value `<= 400` the value is normal
  while `500` and `600` are for medium weights and anything above `700` is `bold`.

  __Font-weight guidance:__ Headings should be between 500-900 while normal text
  should be 300-400. This high contrast in the font-weight will make the headings
  draw more attention.
- _Text Decorative:_ it sets decorative lines on text. The property name is
  `text-decoration` and its values include:
  - _Line value:_ `none`, `underline`, `overline`, `line-through`
  - _Color value:_ `named`, `hex`, `rgb`
  - _Style value:_ `double`, `dotted`, `wavy`, `solid`, `dashed`

  An example of this can be:

  ```css
  text-decoration: underline red dotted;
  ```

  Not recommended to use, unless you want to remove the default `underline` value
  on anchor links.
- _Font Style:_ it sets the style of a font. Its values include: `normal`, `italic`,
  `oblique`, `oblique 10deg`. It can sometimes be used to draw attention.
- _List Style:_ it sets the style of the list. Its values include:
  - `none`: Always us this for structural purposes (e.g. navigation bar)
  - `disc`
  - `circle`
  - `square`
  - `decimal`: This is the default for order list.

For summary card, check out the following asset, page 6 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Text Spacing

- _Text Align:_ specifies the horizontal alignment of text within an element.
  There are 4 values to choose from: `left`, `right`, `center`, and `justify`.
  But note that, only <ins>block</ins> text can move around, because it takes all
  the horizontal space of the screen. You can take a look at this picture
  ![Block-vs-Inline-Text](./assets/Block-vs-Inline-Text-Spacing.png)
  _Text align guidance:_ Don't justify text, long block of text should be left-
  aligned and do not center large blocks of text.
- _Line height:_ sets the height of text and is commonly used to set distance
  between multiple lines of text. The signature is as follows:

  ```css
  // There are several types of values: unitless, percentage, pixels, ems
  line-height: 1.5;
  ```

  Headings should be `< 1.5` and regular text should be `1.5-2` to improve
  readability.
- _Letter spacing:_ sets the horizontal space between characters. The signature
  is

  ```css
  // There are 3 main units that can be used: pixels (most common), percentage, and ems.
  letter-spacing: 8px;
  ```

  We __often__ apply a small negative px value to headings to improve readability,
  which is commonly called _tightening_.

Summary card for this section can be found on page 7 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Text Size

- _Font Size:_ sets the font size of the text. The values can be set to be either
  __static__ or __relative__. Let's first know the signature of the property:

  ```css
  font-size: 16px; // Static value.
  ```

  - __Absolute units:__ size is fixed and does not change in relation to parent
    elements. The units to be used are flexible:
    - px (most commonly used)
    - pt, in, cm, mm
  - __Relative units:__ size is based on the size of the parent element and adjusts
    proportionally to changes in the parent elements. Some units for the values are:
    - %, em, rem, vh, vw

  __Font Size Guidance:__ It is recommended that regular text should be `16px-32px`
  and headings can be `>60px`. We can use a type-scale [⇲](typescale.com)
  which provides a structured hierachy of font sizes to create visual consistency
  and limits choices.

Summary card for this section can be found on page 7 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Font Family

- _Typeface:_ There are 5 types - serif, sans-serif, monospace, cursive, and display.
  - Serif: A classic feel used by brands to communicate __luxury__ and __reliability__.
    They are mostly used by _high-end companies_.
  - Sans-serif: A modern and clean feel used by brands to communicate __simplicity__
    and __clarity__. Thus, they are used by many _tech companies_.
  - Monospace: A technical feel used by brands to communicate __accuracy__ and
    __precision__. Usually, they are used for very technical products.
  - Cursive: A personal feel used to connect with people on a more emotional level.
    Some web pages related to weddings or blogs often use cursive.
- _Font Family:_ sets a prioritized list of font names (typeface) or font categories.

  ```css
  //            1st choice
  font-family: 'Tahoma', sans-serif;
  //                     Fall-back
  ```

  Consider the above piece of code, the way it works is that fonts will only display
  if they are installed on local machines.

  __Font guidance:__ Select 1 or 2 fonts, no more.

  _Google Fonts_ are available online and do not rely on the fonts installed on
  an individual user's device.

Summary card for this section can be found on page 8 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Text Color

- _CSS Color:_ specifies the color of text. Its signature is as follows

  ```css
  color: #9874F9;
  ```

  The value for the `color` property can have different units:
  - hex (most commonly used)
  - rgb, rgba, hsl

  __Color Guidance:__ Have at least 2 colors in your color palette, a primary and
  grey color.
  ![Text-Color-Guidance](./assets/Text-Color-Guidance.png)
  Primary colors are used to highlight important parts of a page (e.g. buttons).
  Tints and shades can be used to create contrast, which are often used for fonts.

  A great place to explore colors and choose you palette is [tailwind colors](https://tailwindcss.com/docs/colors).

Summary card for this section can be found on page 8 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Documentation Example

```css
/// Documentation on properties used in project.
///
/// +----------------------/ Typography /----------------------+
/// |
/// | FONT SIZES:
/// | 12px / 16px / 20px / 24px / 32 px / 40px / 48px / 60px
/// |
/// | FONT WEIGHTS:
/// | 400 / 700
/// |
/// | LINE HEIGHT
/// | 1.2 / 1.5
/// |
/// | LETTER SPACING
/// | 0 / -2px
/// |
/// | FONT FAMILY
/// | Poppins
/// |
/// +----------------------------------------------------------+
///
/// +----------/ COLORS /----------+
/// |
/// | PRIMARY
/// | Base: #339af0 (Sky Blue)
/// | Tint: #d0ebff (Arctic)
/// | Shade: #1c7ed6 (Deep Sea)
/// |
/// | SECONDARY
/// | Base: #9775fa (Lavendar)
/// | Tint: #e5dbff (Moonligh Iris)
/// | Shade: #7048e8 (Velvet Night)
/// |
/// | TERTIARY
/// | Base: #ff922b (Pumpkin)
/// | Tint: #ffe8cc (Dawn)
/// | Shade: #f76707 (Lava)
/// |
/// | GREY
/// | Base: #495057 (Slate)
/// | Tint: #f1f3f5 (Pebble)
/// | Shade: #212529 (Graphite)
/// | White: #ffffff (White)
/// | Black: #000000 (Black)
/// |
/// +------------------------------+
///
/// +---------- BORDER RADIUS ----------+
/// |
/// | 4px / 8px / 20px
/// |
/// +-----------------------------------+
///
/// +----------------------------- SPACING SYSTEM -----------------------------+
/// |
/// | 5px / 10px / 15px / 20px / 25px / 30px / 40px / 50px / 60px / 70px / 80px /
/// | 90px / 100px / 125px / 150px / 200px / 250px / 300px / 400px / 500px
/// |
/// +--------------------------------------------------------------------------+
```

## 🪴 Topic 2: CSS Selectors

### The Type Selectors & The Cascade

- _Type Selector:_ selects elements based on their tag name. For example,

  ```css
  h1 {
    font-size: 20px;
    color: orange;
  }
  ```

  __Type Selector Guidance:__ It is useful for setting global styles to ensure
  consistency.
- _The Cascade:_ styles declared later will take priority. For example,

  ```css
  h1 {
    color: blue;
  }
  h1 {
    color: green; // This is declared later, so this will take effect.
  }
  ```

Summary card for this section can be found on page 10 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Grouping Selectors

- Selectors can be comma separated to apply shared styles.

  ```css
  h1 {
    color: blue;
  }
  h2 {
    color: blue;
  }
  // The above piece of code can be combined into
  h1, h2 {
    color: blue;
  }
  ```

- _Grouping selectos and cascading_

  ```css
  h1 {
    font-size: 62px;
    color: #495057;
  }
  h2 {
    font-size: 48px;
    color: #495057;
  }

  // Since the color property is the same, we can use CSS cascade to write the
  // following piece of code.

  h1, h2 {
    color: #495057;
  }
  h1 {
    font-size: 62px;
  }
  h2 {
    font-size: 48px;
  }
  ```

Summary card for this section can be found on page 10 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### ID & Class Selectors

- _The ID selector_ selects an element based on an uniqe ID attribute and can
  only be used once.
- _The class selector_ selects one or more elements based on a __class__ attribute
  that can be used multiple times.

For example, consider the following html code

```html
<body>
  <p id="red-text"> Copper mug</p>
  <p class="blue-text">Hello</p>
  <p class="blue-text">World</p>
</body>
```

Then, we can use ID and Class selectors to style as follows

```css
p {
  color: yellow;
}
#red-text { // ID selector
  color: red;
}
.blue-text { // Class selector
  color: blue;
}
```

__Selector Guidance:__

- Classes are often preferred over IDs because they offer greater flexibility and
  reusability.
- It is common to use __type selectors__ for global styles and __class selectors__
  for more specific visual styles.
- Classes for components are designed to be combined on a single HTML element for
  a modular approach to styling. For example,

  ```css
  .btn {
    display: inline-block;
    text-decoration: none;
    padding: 1.5rem 3 rem;
    border-radius: 8px;
    font-size: 1.6 rem;
  }
  .btn-primary {}
  .btn-secondary {}
  ```

  Then in html, we can use these class as follows,

  ```html
  <a href="/" class="btn btn-primary">Press Me!</a>
  ```

Summary card for this section can be found on page 10 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Pseudo-class Selectors

Pseudo-classes defines styles for a specific state or condition of an HTML element.
The signature is a keyword with a colon added to the end of a selector.

- __State__ pseudo-classes: dynamic styling based on user interaction & commonly
  used for hyperlinks.
  - `a:link` targets __anchor tags__ that have _not_ yet been visited.
  - `a:visited` targets __anchor tags__ that have been visited.
  - `a:hover` targets an element when the cursor is placed over it.
  - `a:active` targets an element when it is being clicked. However, this pseudo-
    class is not commonly used.

  __State pseudo-classes guidance:__ it is best practice to style the pseudo-classes
  of anchor tags instead of styling the anchor element directly.

  ```css
  // bad practice - does not cover all states.
  a {
    color: orange;
  }
  // good practice
  a:link, a:hover {
    color: orange;
  }
  ```

  Another example of bad-good practice:

  ```css
  // bad practice
  .btn {
    color: purple;
  }
  // good practice
  .btn:link, .btn:visited {
    color: purple;
  }
  ```

- __Conditional__ pseudo-classes: styling based on an element position in relation
  to other elements.
  - `li:first-child` targets the first child element.
  - `li:last-child` targets the last child element.
  - `li:nth-child(n)` targets child elements based on their position.

Summary card for this section can be found on page 11 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Combinator Selectors

Combining two or more selectors to target elements based on their positions
relative to each other.

- `div p` __all__ descendant (child) selector
- `div > p` __direct/first__ descendant (child) selector
- `h1 + p` adjacent (sibling) selector
- `h1 ~ p` General sibling selector

The 1st and 2nd are used the most.

Summary card for this section can be found on page 11 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Specificity

Summary card for this section can be found on page 11 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Inheritance

Properties set on parent elements are passed to their children by default.

_But what gets inherited?_

It is mainly text properties that are inherited from parent to child. Those are
`color`, `font-weight`, `line-height`, `font-family`, `font-style`, `text-align`,
`font-size`, `letter-spacing`, `text-transform`.

__Inheritance & Specificity guidance:__ Global font styles are set on the body element so that
all child text elements inherit styles by default.

```css
body {
  color: black;
  font-weight: 400;
  font-family: Arial, sans-serif;
}

// Override
h1 {
  color: blue;
  font-weight: 700;
  font-size: 52px;
}
```

__Inheritance & Text-Align__

We know that text-align only have effect on block elements, not inline elements.
However, when applying the text-align property to a block-level parent containing
inline elements, all children will be impacted. An illustration of this is as follows,

![Inheritance-and-Text-Align-Before](<./assets/Inheritance-and-Text-Align-Before.png>)

![Inheritance-and-Text-Align-After](<./assets/Inheritance-and-Text-Align-After.png>)

Then the contents inside are also inherited the text-align property from the
container.

![Inheritance-and-Text-Align-Inherited-Before](<./assets/Inheritance-and-Text-Align-Inherited-Before.png>)

![Inheritance-and-Text-Align-Inherited-After](<./assets/Inheritance-and-Text-Align-Inherited-After.png>)

Summary card for this section can be found on page 12 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Pseudo-elements Selectors

Pseudo-elements are used to style a specific part of an element. However, they are
not commonly used.

Summary card for this section can be found on page 12 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

## 🌷 Topic 3: CSS Box Model

All HTML elements are treated as rectangular boxes and each box has it own set of
properties.

![CSS Box Model](<./assets/CSS-Box-model-illustration.png>)

### Background Color

The __background color property__ sets the background color of an element and
applies to the content and any padding. The property name in CSS is
`background-color`.

__Background color guidance:__ It is _sometimes_ used to set the background color
of entire page or _commonly_ used to set the background color of page sections.

_What is the difference between color and background color?_

|          Color           |            Background Color             |
| :----------------------: | :-------------------------------------: |
| Applies to text elements | Applies to the background of an element |
|    `color: #339af0;`     |      `background-color: #339af0;`       |

### Width & Height

#### Default Block-Level Element Dimensions

By default __block-level__ elements are just big enough to fit its contents _vertically_
and stretches full-width _horizontally_. Some examples are `h1`, `p`, etc.

Whereas, by default __inline-level__ elements are just big enough to fits its
contents both vertically and horizontally. Some examples are `a`, `img`, etc.

#### Width & Height Properties

The default box dimensions can be overwritten by the `width` & `height` properties.

```css
div {
  width: 500px;
  height: 100px;
}
```

#### _Block-Level_ Sizing Guidance

Generally, we don't usually set width & height properties of _block-level_ elements.

![Block Level Sizing Guidance](<./assets/CSS-Box-model-wh-block-level-guidance.png>)

![An example](<./assets/CSS-Box-model-wh-block-level-guidance-2.png>)

#### _Inline-Level_ Sizing Guidance

Width and Height can be set on some inline elements but not others.

![An example of inline level sizing](<./assets/CSS-Box-model-wh-inline-level-guidance.png>)

### Padding

__Definition:__ Padding is the space between the content of an element and its border to improve readability and visual design.

There are 2 ways to set the padding property values of an object:

- _Longhand Padding:_ sets the padding on individual sides. E.g.

  ```css
  padding-top: 5px;
  padding-bottom: 5px;
  padding-right: 10 px;
  padding-left: 10 px;
  ```

- _Shorthand Padding:_ sets the padding on all sides at once. E.g.

  ![CSS-Shorthand-padding](<./assets/CSS-Box-model-padding-shorthand.png>)

Summary card for this section can be found on page 14 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Border

___Definition:___ Border creates a visible boundary around an HTML element and can enhance visual appearance and separation from other elements.

There are 3 ways to set the border properties value:

- _Longhand Border Properties:_ sets the individual border properties & applies to
  all four borders.
  - `width`: sets border thickness. It has the following signature

    ```css
    border-width: 5px;
    ```

    The default value is `0px` and the values can be set with `px`, `em`, or `rem`.

  - `style`: sets the border line style. It has the following signature

    ```css
    border-style: none; // DEFAULT
    border-style: solid;
    border-style: dotted;
    border-style: dashed;
    border-style: double;
    ```

  - `color`: sets the border color. It has the following signature

    ```css
    border-color: red;
    ```

    The color can be set using element color or values like `rgb`, `hex`.

- _Shorthand Border:_ sets all the border properties in one go & applies to all
  four borders. The signature is: `border: <width> <style> <color>`. For example,

  ```css
  border: 5px solid red;
  ```

- _Individual Border Properties:_ sets all the border properties in one go & applies to __one__ border at a time.

  ```css
  border-top: 6px solid purple;
  border-bottom: 1px dotted red;
  border-right: 2px dashed blue;
  border-left: 10px double green;
  ```

#### Border Guidance

- Border helps with grouping related contents. E.g. ![CSS-Box-model-border-example-1](<./assets/CSS-Box-model-border-example-1.png>)
- Individual borders help separate sections. E.g. ![CSS-Box-model-border-example-2](<./assets/CSS-Box-model-border-example-2.png>)
- Borders can create outline buttons which are _commonly_ used along side solid filled buttons

#### Border Radius

The border radius property has the following signature,

```css
border-radius: 12px;
```

Values to be set can be in either `px`, `em`, `rem`, or `%`.

<details>
<summary><b>🤔 How border radius work?</b></summary>

__A:__ A quarter-circle is placed in the corner of an element and the cut out rounds
the element corner.

![CSS-Box-model-border-example-3](<./assets/CSS-Box-model-border-example-3.png>)
![CSS-Box-model-border-example-4](<./assets/CSS-Box-model-border-example-4.png>)
</details>

<details>
<summary><b>💊 How to create pills?</b></summary>

__A:__ _Fully-rounded_ corners are created from rectangles by setting the border-radius equal to half of the elements height.

![CSS-Box-model-border-example-5](<./assets/CSS-Box-model-border-example-5.png>)

</details>

<details>
<summary><b>⭕️ How to create circles?</b></summary>

__A:__ Circular elements are created from _squares_ by setting the border radius equal to half of the elements height.

![CSS-Box-model-border-example-6](<./assets/CSS-Box-model-border-example-6.png>)

</details>

<details>
<summary><b>🧺 Border Radius Guidance</b></summary>

- Square corners are more formal.
- Rounding corners can be perceived as more friendly.
- Fully rounding corners can be perceived as playful.
- It's ___important___ to have consistent rounding across all elements.

</details>

<br />

Summary card for this section can be found on page 14 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Box Sizing

The box-sizing property modifies how the total width and height of an element are calculated.

It's not commonly used, and if it's used the common value is

```css
box-sizing: border-box;
```

Summary card for this section can be found on page 15 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Margin

#### Definition

Margin is the space outside of an element's border creating distance between it and neightboring elements.

#### Margin Properties

There are two ways to set the value of the margin properties.

- _Longhand Margin Properties:_ sets the margin on individual sides.

  ![CSS-Box-model-margin-longhand](<./assets/CSS-Box-model-margin-longhand.png>)

- _Shorthand Margin Properties:_

  ![CSS-Box-model-margin-shorthand](<./assets/CSS-Box-model-margin-shorthand.png>)

#### Margin Guidance

1. Use Margin to apply _whitespace_ between groups of elements.
  ![CSS-Box-model-margin-guidance-1](<./assets/CSS-Box-model-margin-guidance-1.png>)
2. Use Margin to apply _whitespace_ between sections.
  ![CSS-Box-model-margin-guidance-2](<./assets/CSS-Box-model-margin-guidance-2.png>)
3. It is common to control spacing between elements with margin and other more modern techniques.

Summary card for this section can be found on page 14 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Display

#### Introduction of Box and Inline Elements

All HTML elements are by default either a block or inline element.
Here are some characteristics of block and inline elements,

|           Block Elements           |            Inline Elements             |
| :--------------------------------: | :------------------------------------: |
|        Starts on a new line        |      Does not start on a new line      |
|   Takes up full width of a page    |  Only occupies width of their content  |
| E.g. `<h1>`, `<p>`, `<ul>`, `<li>` | E.g. `<a>`, `<img>`, `<br>`, `<input>` |
|   ![CSS-Box-model-display-img-1]   |     ![CSS-Box-model-display-img-2]     |

#### _Box and Inline Elements_ vs _The Box Model_

- __Box Elements__: follow _The Box Model_ rules so applying any properties work
  as expected.
- __Inline Elements__: ___DO___ not follow _The Box Model_ rules so applying some
  properties do not work as expected.
  ![CSS-Box-model-display-img-3]

#### Display Property

It sets how the elements is formatted and positioned. There are 5 different values
to be set.

```css
display: block;
display: inline;
display: inline-block;
display: flex;
display: grid;
```

The `inline-block` value is a special value that combines both `inline` and `block` as expected.

![CSS-Box-model-display-img-4]

##### `inline-block` Guidance

It is common to apply `inline-block` to inline elements so they _flow inline_ (i.e. next to each other) but all box properties can be applied. E.g.
![CSS-Box-model-display-inline-block-eg-1]

Summary card for this section can be found on page 15 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### CSS Reset

Browsers have built-in CSS default rules that style HTML elements.

<details>
<summary>
These rules include:
</summary>

- `<body>`
  - `5px` margin
- `<h1>`
  - `21px` margin top & bottom
  - `32px` font-size
  - `700` font-weight
- `<p>`
  - `16px` margin top & bottom
  - `16px` font-size
  - `400` font-weight
- `ol`, `ul`
  - `16px` margin top & bottom
  - `40px` left padding

</details>

<br />

Thus, it is very common to strip away the main default browser styles to give
us a _blank_ canvas to work from.

```css
// Global Resets
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

// Element Resets
a {
  text-decoration: none;
  display: inline-block;
}

ul, ol {
  list-style: none;
}
```

Summary card for this section can be found on page 15 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Main Container

It is very common for a website to have a wrapper which adds margin left & right
as well as centers all the content.
For example,
![CSS-Box-model-container-img-1]

To create a container, a suitable width is chosen to prevent content from excessively
stretching on larger screen size.

#### The keyword `auto`

It's a value that enables the browser to automatically determine a property's size.
E.g.

```css
.img {
  width: 400px;
  height: auto;
}

.container {
  width: 1200px;
  margin-left: auto;
  margin-right: auto;
}
```

Summary card for this section can be found on page 16 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Width

#### Max Width

The maximum width property ensures elements are responsive for different screen sizes.

![CSS-Box-model-container-img-2]

Max-width behaves differently depending on the display property of the HTML element.

![CSS-Box-model-container-img-3]

Here is the comparison between the `width` and `max-width` properties for ___block___ element.

![CSS-Box-model-container-img-4]

For ___inline___ element, it's a little bit different.

![CSS-Box-model-container-img-5]

#### Min Width

It behaves somewhat similar to max width but it doesn't allow the object to go lower
than the `min-width` specified.

#### Width Summary

![CSS-Box-model-container-img-6]

### Height

![CSS-Box-model-container-img-7]

#### Max Height

Max Width and Max Height behave in fundamentally different ways.

- __Max Width__: Based on its container, so it applies even _without_ content.
- __Max Height__: Only applies when _content_ exceeds the limit.

![CSS-Box-model-container-img-8]

This brings us to __overflow__. `Overflow` controls what happens to content that overflows an element box.
Here are the default properties of `overflow`,

```css
overflow: visible;  // Default
overflow: scroll;   // Scrollbar always present
overflow: hidden;   // Clips content
overflow: auto;     // Scrollbar only when necessary
```

<ins><b>Max-Height Guidance</b></ins>:

- Used when you want to ensure elements do not exceed a certain height.

  E.g. the modal window has a max-height & we can use CSS to get a scroll.

  ![CSS-Box-model-container-img-9]

#### Min Height

![CSS-Box-model-container-img-10]

#### Height Summary

![CSS-Box-model-container-img-11]

### Maximum & Minimum Properties

- Max & Min _Width_ are more commonly used compared to Max & Min _Heigth_ due to their role in creating __responsive__ layouts

Summary card for this section can be found on page 16 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

## 👨🏻‍🎨 Topic 4: CSS Units

### Absolute & Relative Units

- Relative units are essential for _responsive_ webpages so elements can dynamically
  adjust for different screen sizes.

<!--region: comparion-->

|                             Absolute Units                              |                                           Relative Units <br /> (Commonly used)                                           |
| :---------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: |
| Size is fixed and does not change in <br /> relation to parent elements | Size is based on the size of a parent <br /> element and adjusts _proportionally_ to <br /> changes in the parent element |
| `px` (Used for specific case) <br /> `pt`, `in`, `cm`, `mm` (Uncommon)  |                                               `%`, `em`, `rem`, `vh`, `vw`                                                |

<!--endregion-->

Summary card for this section can be found on page 18 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Percentages

___Definition:___ A unit that is always relative to some other value.

![CSS-Units-Percentages-img-1]

#### Percentages Guidance

- Percentages are used in conjunction with `max-width` on main containers so that
  the webpage is fully responsive.
- It is common to set images inside a `grid` or `flex` container to `100%` so it
  fills the cell and adapts responsively.
- There are cases when you want to set a button width to 100% so it fill its container
  and adapts responsively.
  ![CSS-Units-Percentages-img-2]
- Percentage values are used for fully rounded corners as pixels require manual
  calculation.
  ![CSS-Units-Percentages-img-3]

Summary card for this section can be found on page 18 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Rems & Ems

__Definition:__

- _`Rems`_: Relative to the root elements font size and are the key ingredient
  for creating responsive webpages. For example,
  ![CSS-Units-Percentages-img-4]

  However, note that by default the root or HTML tag has the font size `16px`.
  If that makes your calculation using `rems` complicated, consider setting the default
  font size or the root font size to `10px`.

  ```css
  html {
    font-size: 10px;
  }
  ```

  <details>
  <summary><b>Rems Guidance</b></summary>

  `rems` are commonly used on `font-size`, `margins`, and `paddings` to create fully responsive webpages.
  ![CSS-Units-Percentages-img-5]

  </details>
- _`Ems`_ (Not as _frequently_ used as `rems`): A relative unit that is more context specific compared to `rems`.

  |                      Typography                      |                   Other Properties                   |
  | :--------------------------------------------------: | :--------------------------------------------------: |
  |     Relative to the font-size <br> of the parent     | Relative to the font-size <br> of the element itself |
  | `font-size` <br> `line-height` <br> `letter-spacing` |  `width` <br> `height` <br> `margin` <br> `padding`  |

  <details>
  <summary><b>Ems Guidance</b></summary>

  Ems provide a higher level of precision for styling smaller components as sizing
  is based on font-size of the element itself.
  ![CSS-Units-Percentages-img-6]

  </details>

### VH & VW

View height (vh) & view width (vw) are units that are a percentage of the browsers
visible window.

![CSS-Units-vh-vw-img-1]

_VH and VW Guidance_:

- `vh` can be used on hero sections in conjunction with `min-height` so content
  always above the fold.
- `vw` can be useful for creating responsive text when it is a main standalone
  element and not confined within a container.

Summary card for this section can be found on page 18 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

## 🧸 Topic 5: CSS Functions

### CSS Variables

(CSS variables) allows us to store values to make it easier to maintain consistenvy
and more easily make global style changes.

![CSS-Functions-CSS-Variables-img-1]

___What is the root pseudo-class?___ A special pseudo-class selector that matches
the root element in a document's hierachy.

![CSS-Functions-CSS-Variables-img-2]

Summary card for this section can be found on page 20 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### CSS Calculations

(CSS Calculations) perform dynamic calculations when setting values.

![CSS-Functions-CSS-Calculations-img-1]

Summary card for this section can be found on page 20 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

## 🤯 Topic 6: Flexbox

### Introduction to Flexbox

#### Definition

Flexbox is a one-dimensional layout model making it simple to arrange items in rows
or columns and distribute space.

#### Flex container and Flex Items

![CSS-Flexbox-img-1]

#### Display `flex`

The `display` property on a container element wil activate flexbox's layout features
on children elements.

![CSS-Flexbox-img-2]

![CSS-Flexbox-img-3]

#### Main and Cross Axis

- The _main axis_ is the primary direction in which flex items are laid out and,
- The _cross axis_ determines distribution on secondary axis.

![CSS-Flexbox-img-4]

#### Flex Direction

It controls the orientation of the main axis.

![CSS-Flexbox-img-5]

<details>
<summary><b>Flex Direction Guidance</b></summary>

- In most cases flex-direction is set to `row` or `column`
  ![CSS-Flexbox-img-6]

</details>

Summary card for this section can be found on page 22 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Normal Flow vs. Flexbox

![CSS-Flexbox-img-7]

#### Normal Flow

- `block` and `inline` elements behave differently in normal flow.
  ![CSS-Flexbox-img-8]
- Applying `text-align` to a block-level parent will impact both `block` & `inline` children through different mechanisms.
  ![CSS-Flexbox-img-9]

#### Flexbox

Flexbox overrides display behavior of `block` & `inline` elements.
![CSS-Flexbox-img-10]

#### Comparison

![CSS-Flexbox-img-11]

Summary card for this section can be found on page 22 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Justify Content

#### Flex Container Size

Understanding a flex-container's dimensions is essential for visualising how flex
items will be positioned.

![CSS-Flexbox-img-12]

#### Justify Content

(It) sets how flex items are positioned along the main axis.

![CSS-Flexbox-img-13]

#### Spacing on The Shifted Main Axis

![CSS-Flexbox-img-14]

#### Justify Content Guidance

- Often used to align and space navigation bars
  ![CSS-Flexbox-img-15]
- Can be used to align entire sections
  ![CSS-Flexbox-img-16]
- Often used for spacing and alignment of simple one-dimensional components
  ![CSS-Flexbox-img-17]
  ![CSS-Flexbox-img-18]

Summary card for this section can be found on page 23 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Align Items

(It) sets how flex items are positioned along the cross axis.

![CSS-Flexbox-img-19]

__Align Items values:__

When no height is set,

![CSS-Flexbox-img-20]

In contrast, when height is set, the flex items have more space to move around.

![CSS-Flexbox-img-21]

#### Shifting the main axis

When no height is set on the container, its height is sum of flex item's height &
`justify` content will have no impact.

![CSS-Flexbox-img-22]

However, if height is set on the container, there can be additional space so justify
content will have an impact.

![CSS-Flexbox-img-23]

#### Align Items Guidance

- Commonly used on navigation menus and footers
  ![CSS-Flexbox-img-24]
- Positioning items inside a component in a flex row
  ![CSS-Flexbox-img-25]
  ![CSS-Flexbox-img-26]

Summary card for this section can be found on page 23 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Gap

#### Flexbox and The Box Model

Box model properties can still be used in a flexbox layout to control spacing.

![CSS-Flexbox-img-27]

#### `Gap` Property

![CSS-Flexbox-img-28]

#### `Margin` and `Gap` Guidance

- `Margin` is used when you want different spacing values between flex items
  ![CSS-Flexbox-img-29]
- `Gap` is used when you want uniform spacing values between flex items
  ![CSS-Flexbox-img-30]

Summary card for this section can be found on page 24 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Nested Flexbox

It is _common_ to have a flexbox layout inside another flexbox layout.
A flex item within a flex container can also function as a flex container itself,
allowing for __multi-level__ flex layouts.

![CSS-Flexbox-img-31]

#### Nested Flexbox Guidance

- Common used in sections and components.
  ![CSS-Flexbox-img-32]

### Centering

Flexbox offers a modern & flexible approach to centering items vertically and
horizontally.

![CSS-Flexbox-img-33]

#### Horizontal Centering a Container

This can be achieved either by using the Box Model `margin` Property or by using
`flexbox`.

![CSS-Flexbox-img-34]

#### Vertical Centering a Container

_(Parent Container Has A Set Height)_

![CSS-Flexbox-img-35]

#### Centering Guidance

__Horizontal Centering Guidance__

- `margin` is often used to establish a webpage's main container as it is a simple
  method for horizontal centering.
  ![CSS-Flexbox-img-36]
- Flexbox properties make it simple to horizontally center regular content containers.
- Padding is used to create equal internal space `left` and `right` around containers like cards & elements like anchor tags.
  ![CSS-Flexbox-img-37]
  This creates an illusion that the ___content___ inside is centered.

__Vertically Centering Guidance__

- Flexbox properties are often used to vertically center UI elements inside a flex container that has a set height.
  ![CSS-Flexbox-img-38]
- Padding is used to create equal internal space `top` and `bottom` around containers like cards & elements like anchor tags.
  ![CSS-Flexbox-img-40]

All in all, here's the table comprises of all the guidance.

![CSS-Flexbox-img-39]

Summary card for this section can be found on page 24 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Flex Children

#### Flex Item Properties

Flexbox also has properties that control the behavior and positioning of flex-items.

![CSS-Flexbox-img-41]

#### Flex-Grow

(It) determines how flex-items expand to fill extra space in a container based on a set
of proportions.

![CSS-Flexbox-img-42]
![CSS-Flexbox-img-43]

#### Flex-Shrink

(It) determines how flex-items shrink relative to others in a container where there
isn't enough space.

![CSS-Flexbox-img-44]

Items shrink based on their `flex-shrink` value relative to the total.

![CSS-Flexbox-img-45]

#### Align-Self

(It) sets individual flex items alignment by overriding the flex containers default
align-items value.

![CSS-Flexbox-img-46]

Here are all the values of `align-self`

![CSS-Flexbox-img-47]

#### Order

Changes the visual order of flex items independent of their order in the HTML
markup.

![CSS-Flexbox-img-48]

Summary card for this section can be found on page 25 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Flex Wrap & Align Content

#### Shrinking Behavior

If flex items are larger than their container, by default they will shrink to fit
inside.

![CSS-Flexbox-img-49]

#### Flex-Wrap

Pushes flex-items onto multiple lines instead of being forced to fit on a single line when they exceed the container.

![CSS-Flexbox-img-50]
![CSS-Flexbox-img-51]

#### Align-Content

Controls the alignment of flex-items along the _cross-axis_ when there are multiple rows of flex-items.

![CSS-Flexbox-img-52]
![CSS-Flexbox-img-53]

#### Guidance

__Flex-Wrap Guidance__ (not often required)

- Wrapping is useful for simple linear layouts when elements need to wrap onto
  multiple lines.

Summary card for this section can be found on page 26 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

## 📺 Topic 7: CSS Grid

### Introduction to CSS Grid

___CSS Grid___ is a two-dimensional layout model making it simple to arrange items
in rows and columns.

#### Terminology

CSS Grid has two main components:

- Grid Container
- Grid Items

![CSS-Grid-Terminology-img-1]

CSS Grid has two axis (can't change direction of the axis) and the grid is
divided into grid cells by grid lines.

![CSS-Grid-Terminology-img-2]

Grid lines define grid tracks (i.e. rows & columns).
Grid rows and columns can have spacing between them.

![CSS-Grid-Terminology-img-3]

#### Display Grid

The display property on a container element will activate grid layout features on
children elements.

```css
display: grid;
```

#### CSS Grid Guidance

- CSS Grid can be used for complex 2D layouts
  ![CSS-Grid-Guidance-img-1]
- Grid is also commonly used for major sections
- Grid is also commonly used for UI element with consistent spacing in a 1D row
  or column
  ![CSS-Grid-Guidance-img-2]

#### CSS Grid vs. Flexbox

![CSS-Grid-Comparison-img-1]

Summary card for this section can be found on page 28 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Grid Columns & Rows

Grid items will appear in a single column by default.

![CSS-Grid-Columns-Rows-img-1]

#### Grid Template Columns

(It) sets the number of columns and width of each.

```css
//                      ↱ width ↰
grid-template-columns: 100px 200px;
//              column-1 ↲     ↳ column-2
```

#### Grid Template Rows

(It) sets the number of rows and height of each.

![CSS-Grid-Columns-Rows-img-2]

#### Fractional Unit

_(It's very commonly used in CSS Grid)_

The fraction of available space in the grid container.

![CSS-Grid-Columns-Rows-img-3]

__Fractional Unit and Rows__

 If height is set on a grid container, `fr` units proportionally distribute space
 to the grid items.

![CSS-Grid-Columns-Rows-img-4]
![CSS-Grid-Columns-Rows-img-5]

#### Repeat Function

Used to repeat rows or columns dimensions when they are recurring.

```css
grid-template-columns: repeat(4, 1fr);

// The upper code will be translated into

grid-template-columns: 1fr 1fr 1fr 1fr;

```

#### Column and Row Guidance

- Columns are often set with `fr` units and the width of grid container is usually
  not set but extends width of the page.
- The number of rows are not usually set but are automatically defined by the number
  of grid items.
- The height of the grid container is not usually set but defined by the sum of the
  height of the grid items.

![CSS-Grid-Columns-Rows-img-6]

Summary card for this section can be found on page 28 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Grid Gap

By default, grid items are placed next to each other with no spacing between them.

#### Gap Property

The `gap` property creates space between grid rows and columns.

![CSS-Grid-Gap-img-1]
![CSS-Grid-Gap-img-2]

#### Gap Guidance

- Commonly used to space grid items in major sections
- Commonly used for spacing in card layouts such as testimonials and features

Summary card for this section can be found on page 28 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Grid _Cell_ Alignment

Grid items can be aligned inside grid cells _if_ there is space available.

#### Justify Items

___(Not `justify-content`)___

![CSS-Grid-Gap-img-3]

#### Align Items

_(Same as flexbox)_

(It) aligns grid items inside a grid cell along the column axis.

![CSS-Grid-Gap-img-4]

#### Alignment Guidance

- It is common to use grid alignment properties when one grid item is larger than
  another.
  ![CSS-Grid-Gap-img-5]
- It is common to use grid and flexbox in parallel to achieve more precise control
  of layout.
  ![CSS-Grid-Gap-img-6]

Summary card for this section can be found on page 29 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Grid _Container_ Alignment

Entire grid tracks can be aligned and distributed when there is extra space in the
grid container.

![CSS-Grid-Container-Alignment-img-1]

#### Justify Content

_(Same as flexbox)_
![CSS-Grid-Container-Alignment-img-2]

#### Align Content

_(Same as flexbox)_
![CSS-Grid-Container-Alignment-img-3]

#### Items vs. Content

![CSS-Grid-Container-Alignment-img-4]

Summary card for this section can be found on page 30 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Grid Items

#### Placing Grid Items

Grid items are automatically placed in a grid based on the order they appear in
the html.

![CSS-Grid-Items-img-1]

Grid items can be moved to different cells.

![CSS-Grid-Items-img-2]
![CSS-Grid-Items-img-3]
![CSS-Grid-Items-img-4]

#### Placing Grid Items Guidance

Placing grid items is used to achieve _specific_ layouts.
![CSS-Grid-Items-img-5]

#### Spanning Grid Cells

`grid-row` and `grid-column` can be used to have grid items span multiple rows or
columns.

![CSS-Grid-Items-img-6]
![CSS-Grid-Items-img-7]

#### Spanning Grid Cells Guidance

Grid cells can be spanned to create visually interesting designs.

![CSS-Grid-Items-img-8]

#### Aligning Grid Items

_(Not too common)_

Grid cell alignment can be overridden for individual items

![CSS-Grid-Items-img-9]

Summary card for this section can be found on page 31 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

## 👟 Topic 8: Responsive Design

Responsive design is powered by three main pillars

1. Relative Units
2. Fluid Layouts
3. Media Queries

### Media Queries

![CSS-Responsive-Media-Queries-img-1]

### Relative Units

#### Rems Inside `@media` Query Conditions

The `rem` value inside a media query condition is always based on the default
browser font size. That means even if we adjust the _browser font-size_ to `10px`
for easier calculation, the condition inside the media query still use `rem` based
on the default `16px`.

```css
:root {
  font-size: 10px; // Set the default font-size to 10px
}

h1 {
  font-size: 80rem; // 80% x 10px = 8px
}

@media (max-width: 31.25rem) { // 31.25% x 16px (not 10px)
  // ...
}
```

#### Rems Inside `@media` Query

Rem values scale proportionally, adjusting both existing and new values based on
the updated root font size.

![CSS-Responsive-Media-Queries-img-2]

When adding new rem values inside a media query select values you would use for
the base case and it will scale automatically.

![CSS-Responsive-Media-Queries-img-3]

#### Fluid Layouts

Using flexbox & CSS grid allow layouts to dynamically adjust in size and position
easily.

![CSS-Responsive-Media-Queries-img-4]

Summary card for this section can be found on page 33 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Breakpoints

__What are breakpoints?__

A specific screen width where a webpage's layout changes.

#### Selecting Breakpoints

Selecting a breakpoints for specific devices is impractical as there are too many to cater for. E.g.,

- Mobile
- Tablet
- Laptop
- Desktop

So knowing the ranges of common device screen sizes is helpful in thinking about
breakpoints.

![CSS-Responsive-Breakpoints-img-1]

Ultimately, selecting breakpoints should be design-led & determined by observing
where the layout naturally 'breaks'.

When using `max-width` in media queries, there's no need to set a maximum range
because styles will be applied automatically.

![CSS-Responsive-Breakpoints-img-2]

Summary card for this section can be found on page 33 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### CSS & Physical Pixels

_Physical Pixels_

A pixel does not have a fixed length and varies between screens.

![CSS-Responsive-Pixels-img-1]

_CSS Pixels_

A CSS pixel has a length of `1/96 inch`

![CSS-Responsive-Pixels-img-2]

Summary card for this section can be found on page 33 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

## 🍓 Topic 9: Positioning

__Normal Flow__

Elements occupy their own designated 'slot' on the page which allows them to be
surrounded by other elements.

![CSS-Positioning-img-1]

### `position` Property

Determines how an element is positioned on the page.

![CSS-Positioning-img-2]

### Static Positioning

The element follows the normal document flow.

```css
position: static; // Default
```

### Relative Positioning

The element follows the normal document flow but can be offset relative to itself.

```css
position: relative;
```

![CSS-Positioning-img-3]
![CSS-Positioning-img-4]

_Offset Properties_ are used to shift element from its reference point.

![CSS-Positioning-img-5]

### Absolute Positioning

The element is removed from the normal document flow.

![CSS-Positioning-img-6]
![CSS-Positioning-img-7]

When positioning an absolute positioned element, the reference point is the root
HTML element.

![CSS-Positioning-img-8]

### Fixed & Sticky Positioning

![CSS-Positioning-img-9]

### Positioning Guidance

- `absolute` positioning is more commonly used than `relative` because it removes
  elements from the document flow.
  ![CSS-Positioning-img-10]
- `absolute` positioning should be used as a last resort for very specific cases
  as it makes responsive design more challenging.
  ![CSS-Positioning-img-11]
- `fixed` positioning is used for elements that need to be absolute positioned but
  also need to remain visible when the user scroll (e.g. the chatbot box).
  ![CSS-Positioning-img-12]
- `sticky` positioning is ideal for navbars, as they stay at the top in normal
  flow and then become fixed when the user scrolls.
  ![CSS-Positioning-img-13]

Summary card for this section can be found on page 35 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Z-index & Stacking Context

#### Z-index

The z-index property controls the vertical stacking order of positioned elements.

![CSS-Z-Index-img-1]
![CSS-Z-Index-img-2]

#### Stacking Context

A positioned element with z-index creates a new stacking context, where its children
are stacked independently of other elements.

![CSS-Z-Index-img-3]

Summary card for this section can be found on page 36 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Transform

Applies visual transformations to an element like scaling, rotating, translating
and skewing.

![CSS-Transform-img-1]

__Transformation Functions__

![CSS-Transform-img-2]

#### Positioned Elements and Translate

Using positioning and translate together provides precise control over element
placement.

![CSS-Transform-img-3]

#### Transform Guidance

- Scale is often used when hovering over an image to create a dynamic visual
  effect.
  ![CSS-Transform-img-4]
- Translate is used on positioned modals to achieve precise centering within their
  containers.
  ![CSS-Transform-img-5]
- Translate is used on positioned alerts to achive precise alignment in top
  right corner.
  ![CSS-Transform-img-6]

Summary card for this section can be found on page 36 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

## ✍🏻 Topic 10: Shadows & Transitions

__How shadows work__

Moving light in the same plane as the table moves the shadow horizontally and
vertically on the floor.

![CSS-Shadows-img-1]

Moving light up and down creates a blur resulting in edges that appear _soft_ or
_sharp_

![CSS-Shadows-img-2]

### Shadow Property

![CSS-Shadows-img-3]
![CSS-Shadows-img-4]

### Shadow Guidance

- Shadows can be used to add depth and dimension to a design to create separation
  between UI elements.
- Shadows should be light, as darker shadows can create a harsh and unnatural
  apprearance in the design.
- Smaller shadows are ideal for enhancing smaller components like buttons and
  cards which add subtle depth.
- Medium-sized shadows can be used for larger components, helping them stand out.
- Large shadows are used when an element should appear to be floating above the rest of the design.

### Flat Design vs. Shadows

- Flat design uses borders and background colors to achieve separation instead of
  shadows.
- Shadows should not be used too often, as overuse can make a design look cluttered
  and overwhelming.

Summary card for this section can be found on page 38 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

### Transitions

When an element moves between states, the changes occur immediately. Transitions
are used to create a smooth, gradual change to an elements properties when it
undergoes a state change.

![CSS-Transitions-img-1]

The `transition` property should be defined on initial state.

![CSS-Transitions-img-2]

#### Timing Functions

Controls the speed of the animation.

![CSS-Transitions-img-3]

#### Transitions Guidance

Transitions can make card components 'pop'.

Summary card for this section can be found on page 38 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

## Resources

- [CSS in 5 minutes](https://www.youtube.com/watch?v=Z4pCqK-V_Wo)
- [Detailed CSS course on Youtube (Part 1)](https://youtu.be/-G-zic_LS0A?si=sooerJXNCADr5Jte)
- [Detailed CSS course on Youtube (Part 2)](https://youtu.be/1ra4yeyjFFc?si=J16lp3yHj1lIzkvx)

<!--Links-->

<!--region: CSS Box Model-->
[CSS-Box-model-display-img-1]: <./assets/CSS-Box-model-display-img-1.png>
[CSS-Box-model-display-img-2]: <./assets/CSS-Box-model-display-img-2.png>
[CSS-Box-model-display-img-3]: <./assets/CSS-Box-model-display-img-3.png>
[CSS-Box-model-display-img-4]: <./assets/CSS-Box-model-display-img-4.png>
[CSS-Box-model-display-inline-block-eg-1]: <./assets/CSS-Box-model-display-inline-block-eg-1.png>
[CSS-Box-model-container-img-1]: <./assets/CSS-Box-model-container-img-1.png>
[CSS-Box-model-container-img-2]: <./assets/CSS-Box-model-container-img-2.png>
[CSS-Box-model-container-img-3]: <./assets/CSS-Box-model-container-img-3.png>
[CSS-Box-model-container-img-4]: <./assets/CSS-Box-model-container-img-4.png>
[CSS-Box-model-container-img-5]: <./assets/CSS-Box-model-container-img-5.png>
[CSS-Box-model-container-img-6]: <./assets/CSS-Box-model-container-img-6.png>
[CSS-Box-model-container-img-7]: <./assets/CSS-Box-model-container-img-7.png>
[CSS-Box-model-container-img-8]: <./assets/CSS-Box-model-container-img-8.png>
[CSS-Box-model-container-img-9]: <./assets/CSS-Box-model-container-img-9.png>
[CSS-Box-model-container-img-10]: <./assets/CSS-Box-model-container-img-10.png>
[CSS-Box-model-container-img-11]: <./assets/CSS-Box-model-container-img-11.png>
<!--endregion-->

<!--region: CSS Units-->
[CSS-Units-Percentages-img-1]: <./assets/CSS-Units-Percentages-img-1.png>
[CSS-Units-Percentages-img-2]: <./assets/CSS-Units-Percentages-img-2.png>
[CSS-Units-Percentages-img-3]: <./assets/CSS-Units-Percentages-img-3.png>
[CSS-Units-Percentages-img-4]: <./assets/CSS-Units-Percentages-img-4.png>
[CSS-Units-Percentages-img-5]: <./assets/CSS-Units-Percentages-img-5.png>
[CSS-Units-Percentages-img-6]: <./assets/CSS-Units-Percentages-img-6.png>
[CSS-Units-vh-vw-img-1]: <./assets/CSS-Units-vh-vw-img-1.png>
<!--endregion-->

<!--region: CSS Functions-->
[CSS-Functions-CSS-Variables-img-1]: <./assets/CSS-Functions-CSS-Variables-img-1.png>
[CSS-Functions-CSS-Variables-img-2]: <./assets/CSS-Functions-CSS-Variables-img-2.png>
[CSS-Functions-CSS-Calculations-img-1]: <./assets/CSS-Functions-CSS-Calculations-img-1.png>
<!--endregion-->

<!--region: CSS Flexbox-->
[CSS-Flexbox-img-1]: <./assets/CSS-Flexbox-img-1.png>
[CSS-Flexbox-img-2]: <./assets/CSS-Flexbox-img-2.png>
[CSS-Flexbox-img-3]: <./assets/CSS-Flexbox-img-3.png>
[CSS-Flexbox-img-4]: <./assets/CSS-Flexbox-img-4.png>
[CSS-Flexbox-img-5]: <./assets/CSS-Flexbox-img-5.png>
[CSS-Flexbox-img-6]: <./assets/CSS-Flexbox-img-6.png>
[CSS-Flexbox-img-7]: <./assets/CSS-Flexbox-img-7.png>
[CSS-Flexbox-img-8]: <./assets/CSS-Flexbox-img-8.png>
[CSS-Flexbox-img-9]: <./assets/CSS-Flexbox-img-9.png>
[CSS-Flexbox-img-10]: <./assets/CSS-Flexbox-img-10.png>
[CSS-Flexbox-img-11]: <./assets/CSS-Flexbox-img-11.png>
[CSS-Flexbox-img-12]: <./assets/CSS-Flexbox-img-12.png>
[CSS-Flexbox-img-13]: <./assets/CSS-Flexbox-img-13.png>
[CSS-Flexbox-img-14]: <./assets/CSS-Flexbox-img-14.png>
[CSS-Flexbox-img-15]: <./assets/CSS-Flexbox-img-15.png>
[CSS-Flexbox-img-16]: <./assets/CSS-Flexbox-img-16.png>
[CSS-Flexbox-img-17]: <./assets/CSS-Flexbox-img-17.png>
[CSS-Flexbox-img-18]: <./assets/CSS-Flexbox-img-18.png>
[CSS-Flexbox-img-19]: <./assets/CSS-Flexbox-img-19.png>
[CSS-Flexbox-img-20]: <./assets/CSS-Flexbox-img-20.png>
[CSS-Flexbox-img-21]: <./assets/CSS-Flexbox-img-21.png>
[CSS-Flexbox-img-22]: <./assets/CSS-Flexbox-img-22.png>
[CSS-Flexbox-img-23]: <./assets/CSS-Flexbox-img-23.png>
[CSS-Flexbox-img-24]: <./assets/CSS-Flexbox-img-24.png>
[CSS-Flexbox-img-25]: <./assets/CSS-Flexbox-img-25.png>
[CSS-Flexbox-img-26]: <./assets/CSS-Flexbox-img-26.png>
[CSS-Flexbox-img-27]: <./assets/CSS-Flexbox-img-27.png>
[CSS-Flexbox-img-28]: <./assets/CSS-Flexbox-img-28.png>
[CSS-Flexbox-img-29]: <./assets/CSS-Flexbox-img-29.png>
[CSS-Flexbox-img-30]: <./assets/CSS-Flexbox-img-30.png>
[CSS-Flexbox-img-31]: <./assets/CSS-Flexbox-img-31.png>
[CSS-Flexbox-img-32]: <./assets/CSS-Flexbox-img-32.png>
[CSS-Flexbox-img-33]: <./assets/CSS-Flexbox-img-33.png>
[CSS-Flexbox-img-34]: <./assets/CSS-Flexbox-img-34.png>
[CSS-Flexbox-img-35]: <./assets/CSS-Flexbox-img-35.png>
[CSS-Flexbox-img-36]: <./assets/CSS-Flexbox-img-36.png>
[CSS-Flexbox-img-37]: <./assets/CSS-Flexbox-img-37.png>
[CSS-Flexbox-img-38]: <./assets/CSS-Flexbox-img-38.png>
[CSS-Flexbox-img-39]: <./assets/CSS-Flexbox-img-39.png>
[CSS-Flexbox-img-40]: <./assets/CSS-Flexbox-img-40.png>
[CSS-Flexbox-img-41]: <./assets/CSS-Flexbox-img-41.png>
[CSS-Flexbox-img-42]: <./assets/CSS-Flexbox-img-42.png>
[CSS-Flexbox-img-43]: <./assets/CSS-Flexbox-img-43.png>
[CSS-Flexbox-img-44]: <./assets/CSS-Flexbox-img-44.png>
[CSS-Flexbox-img-45]: <./assets/CSS-Flexbox-img-45.png>
[CSS-Flexbox-img-46]: <./assets/CSS-Flexbox-img-46.png>
[CSS-Flexbox-img-47]: <./assets/CSS-Flexbox-img-47.png>
[CSS-Flexbox-img-48]: <./assets/CSS-Flexbox-img-48.png>
[CSS-Flexbox-img-49]: <./assets/CSS-Flexbox-img-49.png>
[CSS-Flexbox-img-50]: <./assets/CSS-Flexbox-img-50.png>
[CSS-Flexbox-img-51]: <./assets/CSS-Flexbox-img-51.png>
[CSS-Flexbox-img-52]: <./assets/CSS-Flexbox-img-52.png>
[CSS-Flexbox-img-53]: <./assets/CSS-Flexbox-img-53.png>
<!--endregion-->

<!--region: CSS Grid-->
[CSS-Grid-Terminology-img-1]: <./assets/CSS-Grid-Terminology-img-1.png>
[CSS-Grid-Terminology-img-2]: <./assets/CSS-Grid-Terminology-img-2.png>
[CSS-Grid-Terminology-img-3]: <./assets/CSS-Grid-Terminology-img-3.png>

[CSS-Grid-Guidance-img-1]: <./assets/CSS-Grid-Guidance-img-1.png>
[CSS-Grid-Guidance-img-2]: <./assets/CSS-Grid-Guidance-img-2.png>

[CSS-Grid-Comparison-img-1]: <./assets/CSS-Grid-Comparison-img-1.png>

[CSS-Grid-Columns-Rows-img-1]: <./assets/CSS-Grid-Columns-Rows-img-1.png>
[CSS-Grid-Columns-Rows-img-2]: <./assets/CSS-Grid-Columns-Rows-img-2.png>
[CSS-Grid-Columns-Rows-img-3]: <./assets/CSS-Grid-Columns-Rows-img-3.png>
[CSS-Grid-Columns-Rows-img-4]: <./assets/CSS-Grid-Columns-Rows-img-4.png>
[CSS-Grid-Columns-Rows-img-5]: <./assets/CSS-Grid-Columns-Rows-img-5.png>
[CSS-Grid-Columns-Rows-img-6]: <./assets/CSS-Grid-Columns-Rows-img-6.png>

[CSS-Grid-Gap-img-1]: <./assets/CSS-Grid-Gap-img-1.png>
[CSS-Grid-Gap-img-2]: <./assets/CSS-Grid-Gap-img-2.png>
[CSS-Grid-Gap-img-3]: <./assets/CSS-Grid-Gap-img-3.png>
[CSS-Grid-Gap-img-4]: <./assets/CSS-Grid-Gap-img-4.png>
[CSS-Grid-Gap-img-5]: <./assets/CSS-Grid-Gap-img-5.png>
[CSS-Grid-Gap-img-6]: <./assets/CSS-Grid-Gap-img-6.png>

[CSS-Grid-Container-Alignment-img-1]: <./assets/CSS-Grid-Container-Alignment-img-1.png>
[CSS-Grid-Container-Alignment-img-2]: <./assets/CSS-Grid-Container-Alignment-img-2.png>
[CSS-Grid-Container-Alignment-img-3]: <./assets/CSS-Grid-Container-Alignment-img-3.png>
[CSS-Grid-Container-Alignment-img-4]: <./assets/CSS-Grid-Container-Alignment-img-4.png>

[CSS-Grid-Items-img-1]: <./assets/CSS-Grid-Items-img-1.png>
[CSS-Grid-Items-img-2]: <./assets/CSS-Grid-Items-img-2.png>
[CSS-Grid-Items-img-3]: <./assets/CSS-Grid-Items-img-3.png>
[CSS-Grid-Items-img-4]: <./assets/CSS-Grid-Items-img-4.png>
[CSS-Grid-Items-img-5]: <./assets/CSS-Grid-Items-img-5.png>
[CSS-Grid-Items-img-6]: <./assets/CSS-Grid-Items-img-6.png>
[CSS-Grid-Items-img-7]: <./assets/CSS-Grid-Items-img-7.png>
[CSS-Grid-Items-img-8]: <./assets/CSS-Grid-Items-img-8.png>
[CSS-Grid-Items-img-9]: <./assets/CSS-Grid-Items-img-9.png>
<!--endregion-->

<!--region: Responsive Desgin-->
[CSS-Responsive-Media-Queries-img-1]: <./assets/CSS-Responsive-Media-Queries-img-1.png>
[CSS-Responsive-Media-Queries-img-2]: <./assets/CSS-Responsive-Media-Queries-img-2.png>
[CSS-Responsive-Media-Queries-img-3]: <./assets/CSS-Responsive-Media-Queries-img-3.png>
[CSS-Responsive-Media-Queries-img-4]: <./assets/CSS-Responsive-Media-Queries-img-4.png>

[CSS-Responsive-Breakpoints-img-1]: <./assets/CSS-Responsive-Breakpoints-img-1.png>
[CSS-Responsive-Breakpoints-img-2]: <./assets/CSS-Responsive-Breakpoints-img-2.png>

[CSS-Responsive-Pixels-img-1]: <./assets/CSS-Responsive-Pixels-img-1.png>
[CSS-Responsive-Pixels-img-2]: <./assets/CSS-Responsive-Pixels-img-2.png>
<!--endregion-->

<!--region: Positioning-->
[CSS-Positioning-img-1]: <./assets/CSS-Positioning-img-1.png>
[CSS-Positioning-img-2]: <./assets/CSS-Positioning-img-2.png>
[CSS-Positioning-img-3]: <./assets/CSS-Positioning-img-3.png>
[CSS-Positioning-img-4]: <./assets/CSS-Positioning-img-4.png>
[CSS-Positioning-img-5]: <./assets/CSS-Positioning-img-5.png>
[CSS-Positioning-img-6]: <./assets/CSS-Positioning-img-6.png>
[CSS-Positioning-img-7]: <./assets/CSS-Positioning-img-7.png>
[CSS-Positioning-img-8]: <./assets/CSS-Positioning-img-8.png>
[CSS-Positioning-img-9]: <./assets/CSS-Positioning-img-9.png>
[CSS-Positioning-img-10]: <./assets/CSS-Positioning-img-10.png>
[CSS-Positioning-img-11]: <./assets/CSS-Positioning-img-11.png>
[CSS-Positioning-img-12]: <./assets/CSS-Positioning-img-12.png>
[CSS-Positioning-img-13]: <./assets/CSS-Positioning-img-13.png>
<!--endregion-->

<!--region: z-index & stacking context-->
[CSS-Z-Index-img-1]: <./assets/CSS-Z-Index-img-1.png>
[CSS-Z-Index-img-2]: <./assets/CSS-Z-Index-img-2.png>
[CSS-Z-Index-img-3]: <./assets/CSS-Z-Index-img-3.png>
<!--endregion-->

<!--region: transform-->
[CSS-Transform-img-1]: <./assets/CSS-Transform-img-1.png>
[CSS-Transform-img-2]: <./assets/CSS-Transform-img-2.png>
[CSS-Transform-img-3]: <./assets/CSS-Transform-img-3.png>
[CSS-Transform-img-4]: <./assets/CSS-Transform-img-4.png>
[CSS-Transform-img-5]: <./assets/CSS-Transform-img-5.png>
[CSS-Transform-img-6]: <./assets/CSS-Transform-img-6.png>
<!--endregion-->

<!--region: Shadows & Transitions-->
[CSS-Shadows-img-1]: <./assets/CSS-Shadows-img-1.png>
[CSS-Shadows-img-2]: <./assets/CSS-Shadows-img-2.png>
[CSS-Shadows-img-3]: <./assets/CSS-Shadows-img-3.png>
[CSS-Shadows-img-4]: <./assets/CSS-Shadows-img-4.png>

[CSS-Transitions-img-1]: <./assets/CSS-Transitions-img-1.png>
[CSS-Transitions-img-2]: <./assets/CSS-Transitions-img-2.png>
[CSS-Transitions-img-3]: <./assets/CSS-Transitions-img-3.png>
<!--endregion-->

<!--

Summary card for this section can be found on page 38 [⇲](<./assets/CSS Summary Cards.pdf>)

🚀 [Back to top](#top)

-->
