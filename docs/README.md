# Docs

This is a Next.js documentation site using the [Fumadocs](https://github.com/fuma-nama/fumadocs) framework.

## Local Development

First, install the dependencies:

```bash
npm install
# or
pnpm install
# or
yarn install
```

Then, run the development server:

```bash
npm run dev
# or
pnpm dev
# or
yarn dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Adding Documentation

Adding new documentation pages is simple:

1. Create a new `.mdx` file in the `content/docs` directory
2. Add the page to `content/docs/meta.json` to include it in the navigation

Example `meta.json` structure:

```json
{
  "title": "promptic",
  "root": true,
  "pages": [
    "---Getting Started---", // section headers
    "introduction", // page names
    "---Core Features---",
    "structured-outputs",
    "agents",
    "streaming",
    "error-handling",
    "memory-state",
    "authentication",
    "limitations",
    "---Additional Info---",
    "api-reference"
  ]
}
```


## Deployment

This site can be easily deployed to Vercel given that it's a Next.js app. To do this, you can follow this:

1. Sign up for Vercel
2. Go to [Vercel New Project](https://vercel.com/new/) and import this repository
3. Edit the "Root Directory" field to be `./docs`
4. Click Deploy

After a few minutes it should be deployed at the URL that Vercel provides. Vercel has a generous free tier that should be plenty for a static documentation site.

## Learn More

To learn more about Next.js and Fumadocs, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial
- [Fumadocs](https://fumadocs.vercel.app) - learn about Fumadocs features and configuration

