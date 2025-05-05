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

## Resources

- [CSS in 5 minutes](https://www.youtube.com/watch?v=Z4pCqK-V_Wo)
- [Detailed CSS course on Youtube (Part 1)](https://youtu.be/-G-zic_LS0A?si=sooerJXNCADr5Jte)
- [Detailed CSS course on Youtube (Part 2)](https://youtu.be/1ra4yeyjFFc?si=J16lp3yHj1lIzkvx)
