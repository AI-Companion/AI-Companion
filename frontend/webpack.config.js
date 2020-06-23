const HtmlWebPackPlugin = require("html-webpack-plugin");
const webpack = require('webpack');

const htmlWebpackPlugin = new HtmlWebPackPlugin({
  template: "./src/index.html",
  filename: "./index.html"
});

module.exports = {
  entry: './src/index.jsx',
  module: {
    rules: [
      { test: /\.js$/, exclude: /node_modules/, use: { loader: "babel-loader" } },
      { test: /\.css$/, use: ["style-loader", "css-loader"] },
      { test: /\.jsx$/, use: "babel-loader" }
    ]
  },
  plugins: [htmlWebpackPlugin]
};
