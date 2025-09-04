#!/usr/bin/env python
"""
PyMOL 3D visualization module for AFPDB - similar API to mol3D.py but generates PyMOL sessions

This module provides high-level PyMOL visualization functions that mirror the mol3D.py API,
enabling users to easily generate PyMOL session files (.pse) and images (.png) without
needing to know PyMOL command-line syntax.

Key Features:
- Simple API similar to mol3D.py show() methods
- Automatic PyMOL session (.pse) file generation  
- Optional image (.png) export
- Multiple structures and visualization styles
- Chain-based coloring and styling
- Secondary structure coloring
- B-factor/confidence coloring

Usage:
    >>> from afpdb.pymol3D import PyMOL3D
    >>> viz = PyMOL3D()
    >>> viz.show("protein.pdb", output="visualization.pse")
"""

import os
import tempfile
from string import ascii_uppercase, ascii_lowercase
from .mypymol import PyMOL

# PyMOL color list matching mol3D.py colors
pymol_color_list = [
    "green", "cyan", "magenta", "yellow", "salmon", "gray70", "slate", "orange",
    "palegreen", "teal", "hotpink", "wheat", "violetpurple", "gray50", "marine", "olive",
    "forest", "aquamarine", "brown", "pink", "lightblue", "lightorange", "palecyan", "lightyellow",
    "lightgreen", "lightpink", "lightmagenta", "lightgray", "lightblue", "lightbrown",
    "splitpea", "raspberry", "sand", "smudge", "violet", "chocolate", 
    "silver", "lightcyan", "limon", "slate"
]

# Extended alphabet for chain naming
alphabet_list = list(ascii_uppercase + ascii_lowercase)

class PyMOL3D:
    """PyMOL-based 3D visualization class with fluent interface."""
    
    def __init__(self):
        """Initialize PyMOL3D with a PyMOL instance."""
        self.pymol = None
        self.n_model = 0
        self.model_names = []
        self.chains_per_model = []
        
        # Fluent interface state
        self.output = None
        self.save_png = False
        self.width = 480
        self.height = 480
        
    def _ensure_pymol(self):
        """Ensure PyMOL instance is available."""
        if self.pymol is None:
            self.pymol = PyMOL()
            # Set up basic PyMOL environment
            # self.pymol.run("bg_color white")
            self.pymol.run("set ray_opaque_background, 0")  # Transparent background for PNG
            self.pymol.run("set antialias, 2")  # Better quality
            
    def _get_model_name(self, prefix="model"):
        """Generate unique model name."""
        name = f"{prefix}_{self.n_model}"
        return name
        
    def set_theme(self, theme_name="publication", custom_settings=None):
        """
        Apply PyMOL display settings theme for high-quality visualization.
        
        Args:
            theme_name (str): Predefined theme name. Options:
                - "publication": High-quality publication-ready theme
                - "basic": Simple clean theme
                - "custom": Use custom_settings parameter
            custom_settings (dict or callable): Custom settings to apply.
                Can be a dictionary of PyMOL commands or a callable function.
                
        Returns:
            self: For method chaining
            
        Examples:
            >>> viz = PyMOL3D()
            >>> viz.set_theme("publication")  # Apply publication theme
            >>> viz.show(protein, output="high_quality.pse")
            
            >>> # Custom settings dictionary
            >>> custom = {"stick_radius": 0.3, "cartoon_transparency": 0.2}
            >>> viz.set_theme("custom", custom_settings=custom)
            
            >>> # Custom function
            >>> def my_theme():
            ...     viz.pymol.run("bg_color black")
            ...     viz.pymol.run("set stick_radius, 0.25")
            >>> viz.set_theme("custom", custom_settings=my_theme)
        """
        self._ensure_pymol()
        
        if theme_name == "publication":
            self._apply_publication_theme()
        elif theme_name == "basic":
            self._apply_basic_theme()
        elif theme_name == "custom" and custom_settings:
            self._apply_custom_settings(custom_settings)
        else:
            print(f"Unknown theme '{theme_name}' or missing custom_settings")
            
        return self  # For chaining
        
    def _apply_publication_theme(self):
        """Apply high-quality publication theme based on user's preferences."""
        # Define custom colors
        self.pymol.run('set_color strong_blue, [0.337, 0.443, 0.996]')
        self.pymol.run('set_color off_black, [0.0, 0.0, 0.0]')
        self.pymol.run('set_color off_white, [0.929, 0.937, 0.996]')
        self.pymol.run('set_color purple, [0.7, 0.5, 0.9]')
        self.pymol.run('set_color orange_highlight, [1.0, 0.6, 0.2]')
        self.pymol.run('set_color beige_surface, [0.96, 0.91, 0.80]')
        self.pymol.run('set_color very_weak_purple_blue, [0.733, 0.776, 1.0]')
        self.pymol.run('set_color weak_blue, [0.4, 0.612, 0.914]')
        self.pymol.run('set_color very_weak_blue, [0.74, 0.84, 0.91]')  # Fixed syntax error
        
        #print("Custom colors defined for publication theme")
        
        # Stick and cartoon settings
        self.pymol.run('set stick_radius, 0.20')
        self.pymol.run('set stick_quality, 20')
        self.pymol.run('set cartoon_transparency, 0.0')
        
        # Background and rendering
        self.pymol.run('bg_color white')
        self.pymol.run('set ray_trace_fog, 0')
        self.pymol.run('set antialias, 2')
        self.pymol.run('set depth_cue, 0')
        
        # Use ray tracing for high quality
        self.pymol.run('set ray_trace_color, off_black')
        self.pymol.run('set ray_trace_mode, 1')
        self.pymol.run('set ambient_occlusion_mode, 1')
        self.pymol.run('set ambient_occlusion_scale, 1.0')

        # Line settings
        self.pymol.run('hide lines')
        self.pymol.run('set line_smooth, 1')
        self.pymol.run('set line_width, 1.0')
        self.pymol.run('set ray_trace_gain, 0.1')

        # Surface settings
        self.pymol.run('set surface_quality, 2')
        self.pymol.run('set surface_proximity, 1.8')
        self.pymol.run('set surface_solvent, 0')
        self.pymol.run('set surface_smooth_edges, 1')
        
        # Lighting and shading
        self.pymol.run('set specular, 0.1')
        self.pymol.run('set shininess, 10')
        self.pymol.run('set reflect, 0.02')
        self.pymol.run('set ray_shadows, 1')
        self.pymol.run('set ambient, 0.65')
        self.pymol.run('set direct, 0.50')
        
        # Van der Waals settings
        self.pymol.run('alter all, vdw=3')
        
    def _apply_basic_theme(self):
        """Apply basic clean theme."""
        self.pymol.run('bg_color white')
        self.pymol.run('set antialias, 2')
        self.pymol.run('set stick_radius, 0.15')
        self.pymol.run('set cartoon_transparency, 0.0')
        self.pymol.run('set ray_shadows, 1')
        self.pymol.run('set ambient, 0.5')
        self.pymol.run('set direct, 0.7')
        
    def _apply_custom_settings(self, custom_settings):
        """Apply custom settings from dictionary or callable."""
        if callable(custom_settings):
            # If it's a function, call it
            custom_settings()
        elif isinstance(custom_settings, dict):
            # If it's a dictionary, apply each setting
            for setting, value in custom_settings.items():
                if isinstance(value, str):
                    self.pymol.run(f'set {setting}, {value}')
                else:
                    self.pymol.run(f'set {setting}, {value}')
        
    def reset_theme(self):
        """
        Reset PyMOL to default settings, removing custom themes and colors.
        
        This method restores PyMOL's default rendering settings and removes any
        custom colors or styling applied by themes. Useful for starting fresh
        or switching between different visualization styles.
        
        Returns:
            self: For method chaining
            
        Examples:
            >>> # Apply a theme, then reset to defaults
            >>> viz = PyMOL3D().set_theme("publication").show(protein)
            >>> viz.reset_theme().show(protein2, output="default.pse")
            
            >>> # Reset in the middle of a chain
            >>> (PyMOL3D()
            ...  .set_theme("publication")
            ...  .show(p1, color="purple")
            ...  .reset_theme()
            ...  .show(p2, color="blue")  # Uses default blue
            ...  .show(output="mixed.pse"))
        """
        self._ensure_pymol()       
        # Reset to PyMOL defaults
        self.pymol.run('reinitialize settings')  # This resets most settings to defaults
        return self  # For chaining
    
    def add_model(self, protein, model_name=None):
        """Add a model to PyMOL session."""
        self._ensure_pymol()
        
        if model_name is None:
            model_name = self._get_model_name()
            
        if protein is None:
            n_chains = 1  # Default for pre-loaded models
        else:
            # Load structure into PyMOL
            new_chain_mapping = {old: alphabet_list[i] for i, old in enumerate(protein.chain_id())}
            protein = protein.rename_chains(new_chain_mapping)
            n_chains = len(protein.chain_id())
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
                protein.save(tmp.name)        
            try:
                self.pymol.run(f"load {tmp.name}, {model_name}")
            finally:
                os.unlink(tmp.name)
                
        self.model_names.append(model_name)
        self.chains_per_model.append(n_chains)
        self.n_model += 1
        
        return model_name
        
    def run(self, command):
        """Apply a custom PyMOL command"""
        self.pymol.run(command)
        return self

    def _apply_style(self, model_name, color, style, show_sidechains=False, show_mainchains=False, n_chains=1):
        """Apply visualization style to a model."""
        
        # Clear existing representations
        self.pymol.run(f"hide everything, {model_name}")
        
        # Apply main style
        if style == "cartoon":
            self.pymol.run(f"show cartoon, {model_name}")
        elif style == "stick":
            self.pymol.run(f"show sticks, {model_name}")
        elif style == "sphere":
            self.pymol.run(f"show spheres, {model_name}")
        elif style == "line":
            self.pymol.run(f"show lines, {model_name}")
        elif style == "surface":
            self.pymol.run(f"show surface, {model_name}")
        else:
            # Default to cartoon
            self.pymol.run(f"show cartoon, {model_name}")
            
        # Apply coloring
        if color == "chain":
            # Color each chain differently
            for i in range(min(n_chains, len(pymol_color_list))):
                chain_id = alphabet_list[i]
                pymol_color = pymol_color_list[i]
                self.pymol.run(f"color {pymol_color}, {model_name} and chain {chain_id}")
        elif color in ["lDDT", "b"]:
            # Color by B-factors (confidence/lDDT scores)
            self.pymol.run(f"spectrum b, blue_white_red, {model_name}")
        elif color in ["rainbow", "spectrum"]:
            # Rainbow coloring from N to C terminus
            self.pymol.run(f"spectrum count, rainbow, {model_name}")
        elif color == "ss":
            # Color by secondary structure
            self.pymol.run(f"color red, {model_name} and ss h")  # Helices red
            self.pymol.run(f"color yellow, {model_name} and ss s")  # Sheets yellow  
            self.pymol.run(f"color green, {model_name} and ss l+''")  # Loops green
        else:
            # Single color
            self.pymol.run(f"color {color}, {model_name}")
            
        # Add side chains if requested
        if show_sidechains:
            self.pymol.run(f"show sticks, {model_name} and sidechain")
            self.pymol.run(f"color gray80, {model_name} and sidechain")
            
        # Add main chains if requested  
        if show_mainchains:
            self.pymol.run(f"show sticks, {model_name} and backbone")
            self.pymol.run(f"color gray60, {model_name} and backbone")
            
    def show(self, pdb_data=None, show_sidechains=False, show_mainchains=False, 
             color="chain", style="cartoon", output=None, save_png=None, 
             width=None, height=None, model_name=None):
        """
        Show a protein structure in PyMOL with fluent interface support.
        
        Args:
            pdb_data: Protein data (file path, PDB string, or Protein object).
                     If None, finalizes the visualization and exports files.
            show_sidechains (bool): Show side chains as sticks
            show_mainchains (bool): Show backbone as sticks  
            color (str): Color scheme - "chain", "b", "spectrum", "ss", or color name
            style (str): Display style - "cartoon", "stick", "sphere", "line", "surface"
            output (str): Output .pse file path (can be set/overridden at any point)
            save_png (bool): Whether to also save PNG image (can be set/overridden)
            width (int): PNG image width (can be set/overridden)
            height (int): PNG image height (can be set/overridden)
            model_name (str): Custom model name
            
        Returns:
            self for chaining when pdb_data is provided, or output path when finalizing
            
        Fluent Interface Examples:
            # Chain multiple proteins and finalize
            >>> PyMOL3D().show(p1, color="red").show(p2, color="blue").show(output="result.pse")
            
            # Set output early, override later
            >>> PyMOL3D().show(p1, output="temp.pse").show(p2).show(output="final.pse")
            
            # Use intermediate objects
            >>> viz = PyMOL3D().show(p1, color="red")
            >>> viz.show(p2, color="blue")
            >>> viz.show(output="multi.pse")
        """
        # Update state with any provided parameters
        if output is not None:
            self.output = output
        if save_png is not None:
            self.save_png = save_png
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
            
        # If pdb_data is None, finalize the visualization
        if pdb_data is None:
            if self.n_model > 0:  # Only save if we have models
                if self.output is None:
                    self.output = f"pymol_visualization_{self.n_model}_models.pse"
                self._save_outputs(self.output, self.save_png, self.width, self.height)
                return self.output
            else:
                print("Warning: No models to save")
                return None
        
        # Add the protein model
        self._ensure_pymol()
        
        # Add model
        model_name = self.add_model(pdb_data, model_name)
        n_chains = self.chains_per_model[-1]
        
        # Apply styling
        self._apply_style(model_name, color, style, show_sidechains, 
                         show_mainchains, n_chains)
                
        # Return self for chaining
        return self
        
    def _save_outputs(self, output, save_png, width, height):
        """Save PyMOL session and optional PNG with optimal view."""
        # Optimize view before saving - ensures maximum use of screen space
        self.pymol.run("orient all")  # Orient all objects optimally
        self.pymol.run("zoom complete=1")  # Zoom to include all objects with padding
        self.pymol.run("center all")  # Center all objects in view
        
        # Ensure output has .pse extension
        if not output.endswith('.pse'):
            output = output + '.pse'
            
        # Save PyMOL session
        self.pymol.run(f"save {output}")
        print(f"PyMOL session saved: {output}")
        
        # Save PNG if requested
        if save_png:
            png_output = output.replace('.pse', '.png')
            # Use ray tracing for high-quality images
            self.pymol.run(f"png {png_output}, width={width}, height={height}, dpi=300, ray=1")
            print(f"High-quality image saved: {png_output}")
            
        # Automatically close PyMOL session after saving to free up resources
        self.close()
            
    # Convenience methods matching mol3D.py API
    def cartoon_b(self, pdb_data=None, output=None, save_png=False, width=1200, height=900):
        """Show cartoon representation colored by B-factors."""
        return self.show(pdb_data, color="b", style="cartoon", output=output, 
                        save_png=save_png, width=width, height=height)
        
    def cartoon_spectrum(self, pdb_data=None, output=None, save_png=False, width=1200, height=900):
        """Show cartoon representation with spectrum coloring."""
        return self.show(pdb_data, color="spectrum", style="cartoon", output=output,
                        save_png=save_png, width=width, height=height)
        
    def cartoon_chain(self, pdb_data=None, output=None, save_png=False, width=1200, height=900):
        """Show cartoon representation colored by chain."""
        return self.show(pdb_data, color="chain", style="cartoon", output=output,
                        save_png=save_png, width=width, height=height)
        
    def cartoon_ss(self, pdb_data=None, output=None, save_png=False, width=1200, height=900):
        """Show cartoon representation colored by secondary structure."""
        return self.show(pdb_data, color="ss", style="cartoon", output=output,
                        save_png=save_png, width=width, height=height)
        
    def stick_b(self, pdb_data=None, output=None, save_png=False, width=1200, height=900):
        """Show stick representation colored by B-factors.""" 
        return self.show(pdb_data, color="b", style="stick", output=output,
                        save_png=save_png, width=width, height=height)
        
    def stick_chain(self, pdb_data=None, output=None, save_png=False, width=1200, height=900):
        """Show stick representation colored by chain."""
        return self.show(pdb_data, color="chain", style="stick", output=output,
                        save_png=save_png, width=width, height=height)
        
    def stick_ss(self, pdb_data=None, output=None, save_png=False, width=1200, height=900):
        """Show stick representation colored by secondary structure."""
        return self.show(pdb_data, color="ss", style="stick", output=output,
                        save_png=save_png, width=width, height=height)
        
    def close(self):
        """Close PyMOL session and cleanup."""
        if self.pymol:
            self.pymol.close()
            self.pymol = None
        self.n_model = 0
        self.model_names = []
        self.chains_per_model = []
        
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Convenience function for quick visualization
def quick_show(pdb_data, output="visualization.pse", color="chain", style="cartoon", 
               save_png=False, width=1200, height=900):
    """
    Quick visualization function - create and show structure in one call.
    
    Args:
        pdb_data: Protein data (file path, PDB string, or Protein object)  
        output (str): Output .pse file path
        color (str): Color scheme
        style (str): Display style
        save_png (bool): Whether to also save PNG
        width (int): PNG width
        height (int): PNG height
        
    Returns:
        PyMOL3D: The visualization object (for further manipulation)
    """
    viz = PyMOL3D()
    viz.show(pdb_data, color=color, style=style, output=output, 
             save_png=save_png, width=width, height=height)
    return viz


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        pdb_file = sys.argv[1]
        output_base = pdb_file.replace('.pdb', '')
        
        print("Creating PyMOL visualizations...")
        
        # Create visualization object
        viz = PyMOL3D()
        
        # Show different representations  
        viz.cartoon_chain(pdb_file, output=f"{output_base}_cartoon_chain.pse")
        viz.cartoon_b(pdb_file, output=f"{output_base}_cartoon_b.pse")  
        viz.stick_chain(pdb_file, output=f"{output_base}_stick_chain.pse")
        
        # Multi-model example using fluent interface
        (PyMOL3D()
         .show(pdb_file, color="chain", style="cartoon")
         .show(pdb_file, color="spectrum", style="stick")
         .show(output=f"{output_base}_multi.pse"))
        
        print("Done! Check the generated .pse and .png files.")
    else:
        print("Usage: python pymol3D.py <pdb_file>")
        print("This will generate several example visualizations.")
