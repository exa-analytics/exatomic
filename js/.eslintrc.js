module.exports = {
  parser: '@typescript-eslint/parser',
  env: {
    browser: true,
    es2020: true,
  },
  extends: [
    'eslint:recommended',
    'airbnb-base',
    'plugin:@typescript-eslint/recommended',
  ],
  parserOptions: {
    ecmaVersion: 11,
    sourceType: 'module',
  },
  rules: {
      indent: ['error', 4],
      semi: [2, 'never'],
      'max-classes-per-file': ['error', 4],
      'no-bitwise': ['error', { 'allow': ['>>', '<<', '&'] } ],
      '@typescript-eslint/no-explicit-any': 'off',
  },
  // TODO : typescript indent checking is buggy
  // '@typescript-eslint/indent': ['error', 4],
  overrides: [{
      'files': ['*.ts'],
      'rules': {
          indent: 'off',
      }
  }],
  settings: {
    'import/resolver': {
      node: {
        paths: ['src'],
        extensions: ['.js', '.jsx', '.ts', '.tsx']
      }
    }
  }
}
