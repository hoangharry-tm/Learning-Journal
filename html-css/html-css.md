# 👨🏻‍🎨 HTML/CSS 

_Table of Contents:_

- [✍🏻 Topic 1: CSS Text](#-topic-1--css-text)
  - [Text Styling](#text-styling)

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
  
For summary card, check out the following asset, page 6 [⇲](./assets/CSS Summary Cards.pdf)

### Text Spacing

- _Text Align:_ specifies the horizontal alignment of text within an element.
  There are 4 values to choose from: `left`, `right`, `center`, and `justify`.
  But note that, only <ins>block</ins> text can move around, because it takes all
  the horizontal space of the screen. You can take a look at this picture
  ![Block-vs-Inline-Text](./assets/Block-vs-Inline-Text-Spacing.png)
- 

## Resources

- [CSS in 5 minutes](https://www.youtube.com/watch?v=Z4pCqK-V_Wo)
- [Detailed CSS course on Youtube (Part 1)](https://youtu.be/-G-zic_LS0A?si=sooerJXNCADr5Jte)
- [Detailed CSS course on Youtube (Part 2)](https://youtu.be/1ra4yeyjFFc?si=J16lp3yHj1lIzkvx)
