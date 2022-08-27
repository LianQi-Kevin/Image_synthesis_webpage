prompt_note = """
<details>
<summary>Stable-Diffusion提示词模板</summary>

### 前缀

1. 一幅儿童画：a child's drawing
1. 一张数码照片：a digital painting
1. 油画：an oil painting
1. 抽象画：an abstract drawing
1. 一幅水彩画：A beautiful detailed watercolor painting
1. 一张超现实主义电影感的照片：A cinematic hyperrealism highly detailed photograph
1. 一张浮世绘：A Ukiyo-e print
1. 一幅水墨画：A ink drawing
1. 一张CG级的逼真渲染：A ultra-realistic CG rendering

### 细节

#### 1. 场景
1. 紫禁城：the Forbidden City
1. 哥特风：Gothic style
1. 未来城市：Future city
1. 亭台楼阁：Pavilion
1. 空间站：space station
1. 赛博朋克风格的城市：Cyberpunk city

#### 2. 画面元素
1. 山：mountain
1. 河：river
1. 月：moon
1. 花：flower
1. 阳光：sunshine

#### 3. 结构关系

1. 在什么里面：in
2. 在什么上边：above
3. 有什么：there are/with

#### 4. 颜色描述

1. 两种颜色：black and white
2. 色彩丰富：colorful
3. 单种颜色：yellow color
4. 明亮风格：bright style
5. 暗色系风格：Dark style

#### 5. 参考网站

1. Trending on Facebook
2. Trending on ArtStation
3. Trending on Flickr
4. Trending on Pixiv
5. Trending on CGSociety

#### 6. 艺术家风格

1. Greg Rutkowski
2. Thomas Kinkade
3. Rene Magritte
4. James gurney
5. Vincent van Gogh

#### 7. 描述与控制

1. 视角：
    
   1. 广角 wide angle
   2. 微距 macro
   3. 俯瞰 overlook
   4. 仰视 look up
   5. 超广角 super wide angle
    
2. 画面清晰程度：
   
   1. 精致细节 fine details
   2. 4K
    
3. 画面光影效果:
   1. 光照效果 lighting effect
   2. 体积照明 volume lighting

---

#### 注意事项:

1. 描述语可以切分到多段
1. 慎用动词，尽量使用状态词或形容词代替 例如: `陨石从天而降`换成`一个坠落的陨石`
1. 慎用彩色，可能导致画面过于花哨而缺少美感
1. 尽量选择对的艺术家`Greg Rutkowski`和`Thomas Kinkade`两个默认的画师可以多用
1. 描述语不可过长，否则会报错

</details>

#### 
---
- Model: [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- UI Design: [刘学恺](https://github.com/LianQi-Kevin)

"""

examples = [
    ["cat, sticker, illustration, japanese style", 45, 7.0, 1024, ],
    ["cyberpunk city", 45, 7.0, 1024, ],
    ["A magnificent picture never seen before", 45, 7.0, 1024, ],
    ["what it sees as very very beautiful", 45, 7.0, 1024, ],
    ["a new creature", 45, 7.0, 1024, ],
    ["What does the legendary phoenix  look like", 45, 7.0, 1024, ],
    ["Pug hedgehog hybrid", 45, 7.0, 1024, ],
    ["photo realistic, 4K, ultra high definition, cinematic, sea dragon horse", 45, 7.0, 1024, ],
    ["city of coca cola oil painting", 45, 7.0, 1024, ],
    ["dream come true", 45, 7.0, 1024, ],
    ["Castle in the Sky", 45, 7.0, 1024, ],
    ["AI robot teacher and students kid in classroom ", 45, 7.0, 1024, ],
    ["wasteland, space station, cyberpunk, giant ship, photo realistic, 8K, ultra high definition, cinematic", 45, 7.0, 1024, ],
    ["sunset, sunrays showing through the woods in front, clear night sky, stars visible, mountain in the back, lake in front reflecting the night sky and mountain, photo realistic, 8K, ultra high definition, cinematic", 45, 7.0, 1024, ],
    ["Castle in the sky surrounded by beautiful clouds，photo realistic, 4K, ultra high definition, cinematic", 45, 7.0, 1024, ],
    ["photo realistic, 4K, ultra high definition, cinematic, Castle in the Sky,illustration,chinese style", 45, 7.0, 1024, ],
    ["Castle in the Sky,illustration,chinese style", 45, 7.0, 1024, ],
    ["space soldiers, coming to earth, stars, space ships, purple space", 45, 7.0, 1024, ],
    ['A high tech solarpunk utopia in the Amazon rainforest', 45, 7.5, 1024, ],
    ['A pikachu fine dining with a view to the Eiffel Tower', 45, 7.5, 1024, ],
    ['A mecha robot in a favela in expressionist style', 45, 7.5, 1024, ],
    ['an insect robot preparing a delicious meal', 45, 7.5, 1024, ],
    ["A small cabin on top of a snowy mountain in the style of Disney, artstation", 45, 7.5, 1024, ]
]

parameter_description = """
####
- **Seed**: the seed (for reproducible sampling).
- **Random Seed**: If True, a different seed will be used for each generated image.
- **Steps**: number of ddim sampling steps.
- **Img Height**: image height, in pixel space.
- **Img Width**: image width, in pixel space.
"""
