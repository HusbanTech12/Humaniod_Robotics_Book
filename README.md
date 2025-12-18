# Physical AI & Humanoid Robotics Book

## Overview

This repository contains the complete Physical AI & Humanoid Robotics educational material, implemented as a Docusaurus-based documentation site. The project provides a comprehensive guide and implementation for using ROS 2 (Robot Operating System 2) in humanoid robotics applications.

## Modular Structure

The content is organized into several modules covering different aspects of humanoid robotics:

- **Module 1: The Robotic Nervous System (ROS 2)** - Core ROS 2 concepts and architecture
- **Module 2: The Digital Twin (Gazebo & Unity)** - Simulation and digital twin implementation
- **Module 3: Vision-Language-Action (VLA)** - AI integration for embodied robotics
- **Tutorials** - Practical examples and implementation guides

## Documentation Site

The project includes a Docusaurus-powered documentation site located in the `docs-site/` directory.

### Local Development

1. Navigate to the docs-site directory:
   ```bash
   cd docs-site
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

The documentation will be available at `http://localhost:3000/humanoid-robotic-book/`

### Build for Production

To build the documentation site for production:

```bash
cd docs-site
npm run build
```

## Project Structure

- `docs-site/` - Docusaurus documentation site
- `docs/` - Documentation source files
- `src/` - Source code for examples and ROS 2 packages
- `specs/` - Specification files for different modules
- `simulation_environments/` - Simulation environment configurations
- `unity_projects/` - Unity-based visualization projects

## Deployment

The documentation site can be deployed to multiple platforms:

### GitHub Pages

The site is configured for automatic deployment to GitHub Pages using GitHub Actions.

### Vercel

The site can also be deployed to Vercel with proper configuration.

## Contributing

1. Make changes to the documentation in the `docs-site/docs/` directory
2. Test locally with `npm start`
3. Commit changes to the `main` branch to trigger automatic deployment

## License

This project is licensed under the terms specified in the documentation.