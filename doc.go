// Package convmarkup parses a markup language for
// describing neural networks.
//
// Format overview
//
// This format is line-based, meaning that newlines have
// syntactic significance.
// Lines beginning with a # are treated as comments.
// Empty lines are ignored.
// Every other line either declares a block or closes one.
//
// Everything in the markup language (except for comments)
// is a block.
// A block has a set of numerical attributes and,
// potentially, a set of sub-blocks.
// A markup file is inside an implicit "root block".
//
// Blocks with no children are declared on their own line
// like so:
//
//     BlockName(attr1=val1, attr2=val2, attr3=val3)
//
// As a concrete example, you might declare a layer like
// this:
//
//     Conv(w=3, h=3, n=64, sx=1, sy=1)
//
// If a block has no attributes, the parentheses can be
// omitted, such as in:
//
//     BatchNorm
//     MaxPool(w=2, h=2)
//     ReLU
//
// When a block has children, you use curly braces and
// typically indent the contents of the block:
//
//     Residual {
//         Padding(l=1, r=1, t=1, b=1)
//         Conv(w=3, h=3, n=64)
//     }
//
// Every block takes a tensor as input and produces a
// tensor as output.
// Blocks must be aware of how they manipulate tensor
// dimensions.
// To tell the first block about its input dimensions,
// every file must begin with an Input block:
//
//     Input(w=224, h=224, d=3)
//
// Block types
//
// There are a number of built-in block types for creating
// convolutional neural networks.
// This section describes each one and its attributes.
//
// The Input block should be the first block in every file
// and determines the input tensor dimensions.
// It has three attributes: w for width, h for height, and
// d for depth.
//
// The Conv block defines a convolutional layer.
// The w and h attributes control filter width and height.
// The n attribute controls the number of filters.
// Optional sx and sy attributes determine the x and y
// strides.
// Absent strides are assumed to be 1.
//
// The MaxPool block defines a max-pooling layer.
// The w and h attributes set the pool width and height,
// and sx and sy set the pool stride.
// If a stride is not specified, it will be defaulted to
// the corresponding span for that dimension.
// Max-pooling layers drop partial pools.
//
// The MeanPool block defines a mean-pooling layer and
// works the same way as MaxPool.
//
// The BatchNorm block is a batch normalization layer.
//
// The ReLU block is a ReLU activation layer.
//
// The Sigmoid block is a sigmoid activation layer.
//
// The Tanh block is a tanh activation layer.
//
// The Softmax block is a Softmax activation layer.
//
// The Padding block performs tensor padding.
// It has four attributes, t, b, r, l, for top, bottom,
// right, and left padding respectively.
//
// The Resize block uses some form of interpolation to
// change the width and height of the input tensor.
// It has two attributes, w and h, for width and height
// respectively.
// Neither the input to nor output from a Resize may be
// empty.
//
// The Residual block defines a residual grouping.
// A Residual's sub-blocks are combined and used as the
// residual mapping.
// If the residual mapping needs to do a projection to
// resize the input, an optional Projection block can be
// used:
//
//     Residual {
//         Projection {
//             Conv(w=1, h=1, n=64)
//         }
//         Padding(l=1, r=1, t=1, b=1)
//         Conv(w=3, h=3, n=64)
//     }
//
// The Assert block has no effect besides ensuring that
// the input dimensions are specific values.
// Like Input, it has three attributes: w, h, and d.
//
// The FC block defines a fully-connected layer.
// It has one attribute: "out", which determines the
// output count.
// The output tensor of an FC has a width and height of 1.
//
// The Repeat block repeats its sub-blocks a given number
// of times.
// The n attribute specifies the total number of copies.
// The output dimension must be equal to the input
// dimension, even with n=1.
// An n value of zero is invalid.
//
// The Linear block scales input components by a constant
// "scale" and then adds another constant, "bias".
// If scale or bias are absent, they default to 1 and 0
// respectively.
//
// The Dropout block randomly eliminates a fraction of the
// values.
// The prob attribute, which is required, specifies the
// probability of keeping a value.
package convmarkup
