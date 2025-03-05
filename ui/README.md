# Rant to Rock UI

This is the frontend UI for the Rant to Rock application, which transforms audio recordings or text transcripts into semantically organized Obsidian mind maps.

## Features

- Upload audio recordings or text transcripts
- Monitor processing progress in real-time
- Preview generated clusters and topic summaries
- Download the final Obsidian-compatible ZIP archive

## Tech Stack

- React with TypeScript
- Vite for build tooling
- TailwindCSS for styling
- React Router for navigation
- React Query for data fetching

## Getting Started

### Prerequisites

- Node.js 16+ and npm/yarn/bun
- Backend API server running (see main project README)

### Environment Setup

Create a `.env.local` file in the root of the UI directory with the following content:

```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Adjust the URL if your backend is running on a different host or port.

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at http://localhost:3000.

## Project Structure

```
ui/
├── public/           # Static assets
├── src/              # Source code
│   ├── components/   # React components
│   │   ├── ui/       # UI components (buttons, inputs, etc.)
│   │   ├── FileUploader.tsx    # File upload component
│   │   ├── ProcessingStatus.tsx # Processing status component
│   │   └── PreviewSection.tsx  # Results preview component
│   ├── hooks/        # Custom React hooks
│   │   └── useFileProcessing.tsx # File processing hook
│   ├── lib/          # Utility functions
│   ├── pages/        # Page components
│   │   ├── Index.tsx # Main page
│   │   └── NotFound.tsx # 404 page
│   ├── types/        # TypeScript type definitions
│   ├── App.tsx       # Main application component
│   └── main.tsx      # Application entry point
├── .env.local        # Environment variables
├── package.json      # Dependencies and scripts
└── README.md         # This file
```

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint
- `npm run format` - Format code with Prettier

### Connecting to the Backend

The UI communicates with the backend API using the `useFileProcessing` hook. The hook handles:

1. File uploads via `POST /api/upload`
2. Status polling via `GET /api/status`
3. Fetching preview data via `GET /api/cluster` and `GET /api/summarize`
4. Downloading the ZIP archive via `GET /api/export/zip`

Make sure the backend server is running before starting the UI development server.

## Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory and can be served by any static file server.

## License

See the main project repository for license information.
