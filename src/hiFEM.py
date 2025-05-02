from confmap.confmap import BFF, CETM
import pyvista as pv
import os, time
import numpy as np
import plotly.graph_objects as go
import plotly.graph_objects as go
from scipy.sparse import coo_matrix, csr_matrix, identity
from scipy.sparse.linalg import spsolve
import trame
from pathlib import Path
from typing import Union, Optional, Tuple


def plot_faces_localCSYS(r3faces, igfaces, unfolded, e1_r3, e2_r3, normal_r3, e1_ig, e2_ig):
    """This helper plot function is set only for debudding and works only when the number of faces is 1.
    See the first usage (Ctrl+D) to see how it is used.
    """
    # r3face figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=r3faces[0, [0, 1, 2, 0], 0],
            y=r3faces[0, [0, 1, 2, 0], 1],
            z=r3faces[0, [0, 1, 2, 0], 2],
            mode="lines",
            name="r3faces",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[r3faces[0, 0, 0], r3faces[0, 1, 0], r3faces[0, 2, 0]],
            y=[r3faces[0, 0, 1], r3faces[0, 1, 1], r3faces[0, 2, 1]],
            z=[r3faces[0, 0, 2], r3faces[0, 1, 2], r3faces[0, 2, 2]],
            mode="text",
            text=["i", "j", "k"],
        )
    )
    for vec, name in zip((e1_r3, e2_r3, normal_r3), ("e1_r3", "e2_r3", "normal_r3")):
        fig.add_trace(
            go.Scatter3d(
                x=[r3faces[0, 0, 0] + vec[0, 0], r3faces[0, 0, 0]],
                y=[r3faces[0, 0, 1] + vec[0, 1], r3faces[0, 0, 1]],
                z=[r3faces[0, 0, 2] + vec[0, 2], r3faces[0, 0, 2]],
                mode="lines",
                line=dict(dash="dash"),
                name=name,
            )
        )
    fig.update_layout(scene=dict(dragmode="orbit"))
    fig.write_html("outputs/r3face.html", auto_open=True)

    # igface figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=igfaces[0, [0, 1, 2, 0], 0],
            y=igfaces[0, [0, 1, 2, 0], 1],
            z=[0, 0, 0, 0],
            mode="lines",
            name="igfaces",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=unfolded[0, [0, 1, 2, 0], 0],
            y=unfolded[0, [0, 1, 2, 0], 1],
            z=[0, 0, 0, 0],
            mode="lines",
            name="unfolded",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[igfaces[0, 0, 0], igfaces[0, 1, 0], igfaces[0, 2, 0]],
            y=[igfaces[0, 0, 1], igfaces[0, 1, 1], igfaces[0, 2, 1]],
            z=[0, 0, 0],
            mode="text",
            text=["i", "j", "k"],
        )
    )
    for vec, name in zip((e1_ig, e2_ig), ("e1_ig", "e2_ig")):
        fig.add_trace(
            go.Scatter3d(
                x=[igfaces[0, 0, 0] + vec[0, 0], igfaces[0, 0, 0]],
                y=[igfaces[0, 0, 1] + vec[0, 1], igfaces[0, 0, 1]],
                z=[0, 0],
                mode="lines",
                line=dict(dash="dash"),
                name=name,
            )
        )
    fig.update_layout(scene=dict(dragmode="orbit"))
    fig.write_html("outputs/igface.html", auto_open=True)

    ### All faces in one space with node labels
    # For igfaces and unfolded, add a zero z-coordinate.
    face_index = 0
    ig_face = np.hstack([igfaces[face_index], np.zeros((igfaces[face_index].shape[0], 1))])
    ig_face = np.vstack([ig_face, ig_face[0]])

    unfolded_face = np.hstack([unfolded[face_index], np.zeros((unfolded[face_index].shape[0], 1))])
    unfolded_face = np.vstack([unfolded_face, unfolded_face[0]])

    # For the 3D face, close the loop and align its centroid with ig_face.
    r3_face = np.vstack([r3faces[face_index], r3faces[face_index][0]])
    r3_face += ig_face.mean(axis=0) - r3_face.mean(axis=0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=ig_face[:, 0],
            y=ig_face[:, 1],
            z=ig_face[:, 2],
            mode="lines",
            name="IG Face",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=unfolded_face[:, 0],
            y=unfolded_face[:, 1],
            z=unfolded_face[:, 2],
            mode="lines",
            name="Unfolded Face",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=r3_face[:, 0],
            y=r3_face[:, 1],
            z=r3_face[:, 2],
            mode="lines",
            name="R3 Face",
        )
    )
    labels = ["i", "j", "k"]
    fig.add_trace(
        go.Scatter3d(
            x=ig_face[:-1, 0],
            y=ig_face[:-1, 1],
            z=ig_face[:-1, 2],
            mode="markers+text",
            text=labels,
            textposition="top center",
            name="IG Nodes",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=unfolded_face[:-1, 0],
            y=unfolded_face[:-1, 1],
            z=unfolded_face[:-1, 2],
            mode="markers+text",
            text=labels,
            textposition="top center",
            name="Unfolded Nodes",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=r3_face[:-1, 0],
            y=r3_face[:-1, 1],
            z=r3_face[:-1, 2],
            mode="markers+text",
            text=labels,
            textposition="top center",
            name="R3 Nodes",
        )
    )
    fig.update_layout(scene=dict(dragmode="orbit"))
    fig.write_html("outputs/faces.html", auto_open=True)


def align_surfaces(verts1, verts2, faces, offset=0.0):
    """
    Args:
        verts1 (np.ndarray): vertices of the first surface
        verts2 (np.ndarray): vertices of the second surface
        faces  (np.ndarray): faces of both surfaces. (both surfaces must have the same faces and conencivity)
    Returns:
        np.ndarray: verts1 vertices after alignment

    1. **Preprocessing of Vertices**
    - **Convert to 3D (if necessary):**
        If any surface (e.g., surf1) is given in 2D, extend the vertices by appending a zero for the z-coordinate so that all points are represented in 3D.

    2. **Compute SVD on Both Surfaces**
    - **SVD Decomposition:**
        For both surfaces, perform Singular Value Decomposition (SVD) on the vertex set. This yields three orthogonal singular vectors for each surface that form a local coordinate system capturing the primary geometric directions.

    3. **Adjust the SVD Coordinate System of Surf2 Using the Average Normal**
    - **Average Normal Calculation:**
        Compute normals for each face of surf2 (using cross products of edges) and average these to obtain a consistent overall normal.
    - **SVD Z-Axis Correction:**
        Compare the third singular vector (local z-axis) of surf2’s SVD with this average normal. If the dot product is negative (indicating an opposite direction), flip the z-axis (and adjust the other axes as needed to maintain a right-handed system).

    4. **Select a Common Anchor Face**
    - **Shared Connectivity Advantage:**
        Since both surfaces share the same face connectivity, select the anchor face once based on the topology.
    - **Anchor Face Choice:**
        For example, choose the face with the highest local z-coordinate in surf2’s adjusted coordinate system. This face serves as the common reference for both surfaces.

    5. **Refine SVD Coordinate Systems with Anchor Face Features**
    - **Extract Key Features:**
        From the chosen anchor face, extract its normal and the first edge vector.
    - **Sign Correction for Both Surfaces:**
        Use these vectors to adjust (or “flip”) the SVD-derived coordinate systems of both surf1 and surf2, ensuring that they align consistently with the anchor face’s orientation.

    6. **Transform and Align the Surfaces**
    - **Local Transformation of Surf1:**
        Convert the vertices of surf1 into its own adjusted SVD coordinate system (surf1_local_sys).
    - **Rotation into Surf2’s Local System:**
        Express the points from surf1_local_sys in the coordinate system of surf2 (surf2_local_sys). This effectively rotates the vertices of surf1 into the orientation of surf2.
    - **Global Translation Using Anchor Face Center:**
        Calculate the center (centroid) of the anchor face in surf2. Then translate the transformed surf1 vertices so that the anchor face center of surf1 aligns exactly with the anchor face center of surf2 in global coordinates.

    7. **Output the Aligned Vertices**
    - **Final Result:**
        The output is the set of surf1 vertices that have been rotated and translated, such that they are fully aligned with surf2 using both the adjusted local coordinate systems and the common anchor face reference.

    ### Summary
    - **3D Consistency:** Both surfaces are ensured to be in 3D.
    - **Robust Local Systems:** SVD provides the principal axes, adjusted using the average normal (for surf2) and refined with anchor face features.
    - **Shared Reference:** A single, common anchor face (given shared connectivity) is used for robust alignment.
    - **Two-Step Transformation:**
    1. Rotate surf1 vertices from their own SVD system into surf2’s local system.
    2. Translate these rotated vertices to match the global position of surf2’s anchor face center.
    This strategy guarantees that not only the orientations match (via the local coordinate system conversion) but also that the critical reference point (the anchor face center) is precisely aligned between the two surfaces.
    """
    # --- 1. Ensure vertices are 3D ---
    if verts1.shape[1] == 2:
        verts1 = np.hstack([verts1, np.zeros((verts1.shape[0], 1))])
    if verts2.shape[1] == 2:
        verts2 = np.hstack([verts2, np.zeros((verts2.shape[0], 1))])

    # --- 2. Compute SVD on both surfaces ---
    # Center the vertices
    center1 = np.mean(verts1, axis=0)
    center2 = np.mean(verts2, axis=0)
    X1 = verts1 - center1
    X2 = verts2 - center2
    # SVD: note that np.linalg.svd returns U, S, Vt so that V = Vt.T
    _, _, Vt1 = np.linalg.svd(X1, full_matrices=False)
    R1 = Vt1.T  # SVD basis for surf1 (columns: principal axes)
    _, _, Vt2 = np.linalg.svd(X2, full_matrices=False)
    R2 = Vt2.T  # SVD basis for surf2

    # --- 3. Adjust surf2 SVD using average normal ---
    # Compute normals for all faces of surf2 using vectorized cross product.
    v0 = verts2[faces[:, 0]]
    v1_ = verts2[faces[:, 1]]
    v2 = verts2[faces[:, 2]]
    face_normals = np.cross(v1_ - v0, v2 - v0)
    # Normalize each face normal
    norm_lengths = np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8
    face_normals /= norm_lengths
    # Average normal across faces and normalize it
    avg_normal = np.mean(face_normals, axis=0)
    avg_normal /= np.linalg.norm(avg_normal) + 1e-8
    # If surf2’s third axis is opposite to avg_normal, flip it.
    if np.dot(R2[:, 2], avg_normal) < 0:
        R2[:, 2] = -R2[:, 2]
        # Recompute the other axes to keep a right-handed system.
        R2[:, 1] = np.cross(R2[:, 2], R2[:, 0])
        R2[:, 0] = np.cross(R2[:, 1], R2[:, 2])

    # --- 4. Select a common anchor face ---
    # Express surf2 vertices in its SVD coordinate system.
    local_coords2 = X2.dot(R2)  # shape (n_points, 3)
    # For each face, compute the average local z-coordinate.
    # (faces is (n_faces, 3) so local_coords2[faces] has shape (n_faces, 3, 3))
    face_z = np.mean(local_coords2[faces][:, :, 2], axis=1)
    anchor_idx = np.argmax(face_z)
    anchor_face = faces[anchor_idx]

    # --- 5. Refine SVD coordinate systems using anchor face features ---
    # Extract anchor face vertices from surf2 (global coordinates)
    a0_2, a1_2, a2_2 = verts2[anchor_face[0]], verts2[anchor_face[1]], verts2[anchor_face[2]]
    # Compute the face normal from the anchor face on surf2 and normalize it.
    anchor_normal2 = np.cross(a1_2 - a0_2, a2_2 - a0_2)
    anchor_normal2 /= np.linalg.norm(anchor_normal2) + 1e-8
    # Choose the first edge (from a0 to a1) on surf2 and normalize.
    anchor_edge2 = a1_2 - a0_2
    anchor_edge2 /= np.linalg.norm(anchor_edge2) + 1e-8

    # Adjust surf2’s SVD basis to be aligned with its *own* anchor face features:
    if np.dot(R2[:, 0], anchor_edge2) < 0:
        R2[:, 0] = -R2[:, 0]
    if np.dot(R2[:, 2], anchor_normal2) < 0:
        R2[:, 2] = -R2[:, 2]
    # Recompute second axis to preserve a right-handed coordinate system.
    R2[:, 1] = np.cross(R2[:, 2], R2[:, 0])
    R2[:, 1] /= np.linalg.norm(R2[:, 1]) + 1e-8  # Ensure normalization

    # Extract anchor face vertices from surf1 (global coordinates)
    a0_1, a1_1, a2_1 = verts1[anchor_face[0]], verts1[anchor_face[1]], verts1[anchor_face[2]]
    # Compute the face normal from the anchor face on surf1 and normalize it.
    anchor_normal1 = np.cross(a1_1 - a0_1, a2_1 - a0_1)
    anchor_normal1 /= np.linalg.norm(anchor_normal1) + 1e-8
    # Choose the first edge (from a0 to a1) on surf1 and normalize.
    anchor_edge1 = a1_1 - a0_1
    anchor_edge1 /= np.linalg.norm(anchor_edge1) + 1e-8

    # Adjust surf1’s SVD basis to be aligned with its *own* anchor face features:
    if np.dot(R1[:, 0], anchor_edge1) < 0:
        R1[:, 0] = -R1[:, 0]
    if np.dot(R1[:, 2], anchor_normal1) < 0:
        R1[:, 2] = -R1[:, 2]
    # Recompute second axis to preserve a right-handed coordinate system.
    R1[:, 1] = np.cross(R1[:, 2], R1[:, 0])
    R1[:, 1] /= np.linalg.norm(R1[:, 1]) + 1e-8  # Ensure normalization

    # --- 6. Transform and align the surfaces ---
    # Compute the rotation matrix mapping surf1’s SVD basis into surf2’s.
    R = R2.dot(R1.T)
    # Compute the anchor face centroids for both surfaces.
    anchor_centroid2 = np.mean(verts2[anchor_face], axis=0)
    anchor_centroid1 = np.mean(verts1[anchor_face], axis=0)
    # Rotate all surf1 vertices and then compute the translation needed to align the anchor centroids.
    aligned_verts1 = (R.dot(verts1.T)).T
    translation = anchor_centroid2 - R.dot(anchor_centroid1)
    aligned_verts1 += translation

    # --- 7. 4mm additional translation to ensure the surfaces are not overlapping ---
    aligned_verts1 += offset * avg_normal

    return aligned_verts1


def match_surface_area(verts, faces, uv_verts):
    """Scales uv_verts to match the surface area of the original 3D mesh
    and translates the result so that the minimum coordinate is (0,0).

    Args:
        verts (ndarray): numverts x 3 array of 3D vertices.
        faces (ndarray): numfaces x 3 array of triangle indices.
        uv_verts (ndarray): numverts x 2 array of UV vertices.

    Returns:
        ndarray: Scaled and translated UV vertices.
    """
    # Vectorized computation for 3D triangle areas.
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross_products = np.cross(v1 - v0, v2 - v0)
    areas_3d = 0.5 * np.linalg.norm(cross_products, axis=1)
    total_area_3d = areas_3d.sum()

    # Vectorized computation for 2D (UV) triangle areas.
    uv0 = uv_verts[faces[:, 0]]
    uv1 = uv_verts[faces[:, 1]]
    uv2 = uv_verts[faces[:, 2]]
    # Compute the determinant for each triangle.
    det = (uv1[:, 0] - uv0[:, 0]) * (uv2[:, 1] - uv0[:, 1]) - (uv1[:, 1] - uv0[:, 1]) * (uv2[:, 0] - uv0[:, 0])
    areas_uv = 0.5 * np.abs(det)
    total_area_uv = areas_uv.sum()

    # Compute the scaling factor to match the areas.
    scale_factor = np.sqrt(total_area_3d / total_area_uv)
    uv_scaled = uv_verts * scale_factor

    # Translate the scaled UV coordinates so that the minimum point is at (0, 0).
    uv_translated = uv_scaled - uv_scaled.min(axis=0)

    return uv_translated


def unfold_faces(r3faces, igfaces):
    """
    Unfolds each face by mapping its intrinsic 3D geometry (r3faces) into the 2D domain
    defined by the initial guess (igfaces). This is done by computing a local coordinate system
    for each face (using the face centroid and a basis defined by the first edge and its perpendicular)
    and then re-projecting the intrinsic (local) 3D coordinates into the global 2D coordinates of igfaces.

    Parameters:
        r3faces (np.ndarray): (nfaces, 3, 3) array representing faces in 3D.
        igfaces (np.ndarray): (nfaces, 3, 2) array representing corresponding 2D faces.

    Returns:
        np.ndarray: unfolded faces as an array of shape (nfaces, 3, 2)
    """
    # Compute centroids for both 3D faces and 2D initial guess faces.
    centroid_r3 = r3faces.mean(axis=1)  # shape: (nfaces, 3)
    centroid_ig = igfaces.mean(axis=1)  # shape: (nfaces, 2)

    # For the 3D face, define e1 as the normalized vector along the first edge.
    e1_r3 = r3faces[:, 1] - r3faces[:, 0]  # shape: (nfaces, 3)
    e1_r3 = e1_r3 / np.linalg.norm(e1_r3, axis=1, keepdims=True)

    # For the 2D face, define e1 similarly.
    e1_ig = igfaces[:, 1] - igfaces[:, 0]  # shape: (nfaces, 2)
    e1_ig = e1_ig / np.linalg.norm(e1_ig, axis=1, keepdims=True)

    # Compute the normal for each 3D face (using two edges).
    v0 = r3faces[:, 0]
    v1 = r3faces[:, 1]
    v2 = r3faces[:, 2]
    normal_r3 = np.cross(v1 - v0, v2 - v0)  # shape: (nfaces, 3)
    normal_r3 = normal_r3 / np.linalg.norm(normal_r3, axis=1, keepdims=True)

    # Compute the second basis vector for 3D faces: e2 is perpendicular to e1_r3 in the plane.
    e2_r3 = np.cross(normal_r3, e1_r3)  # shape: (nfaces, 3)
    e2_r3 = e2_r3 / np.linalg.norm(e2_r3, axis=1, keepdims=True)

    # For the 2D faces, the perpendicular of e1 is given by rotating 90°.
    # If e1 = (a, b), then e2 = (-b, a).
    e2_ig = np.stack([-e1_ig[:, 1], e1_ig[:, 0]], axis=1)  # shape: (nfaces, 2)
    # Note: Since e1_ig is normalized, e2_ig is automatically normalized.

    # Compute the local 2D coordinates for each 3D face.
    # For each vertex, the local coordinates are:
    #   local_x = dot( (v - centroid_r3), e1_r3 )
    #   local_y = dot( (v - centroid_r3), e2_r3 )
    r3_shifted = r3faces - centroid_r3[:, None, :]  # shape: (nfaces, 3, 3)
    local_x = np.sum(r3_shifted * e1_r3[:, None, :], axis=2)  # shape: (nfaces, 3)
    local_y = np.sum(r3_shifted * e2_r3[:, None, :], axis=2)  # shape: (nfaces, 3)

    # Stack to form local coordinates for each vertex (nfaces, 3, 2)
    local_r3 = np.stack([local_x, local_y], axis=2)

    # Now, reconstruct the unfolded face in the global 2D domain defined by igfaces.
    # For each face, the unfolded vertex is computed as:
    #   unfolded = centroid_ig + (local_r3_x * e1_ig) + (local_r3_y * e2_ig)
    unfolded = centroid_ig[:, None, :] + local_r3[:, :, 0][:, :, None] * e1_ig[:, None, :] + local_r3[:, :, 1][:, :, None] * e2_ig[:, None, :]

    if r3faces.shape[0] == 1:
        plot_faces_localCSYS(r3faces, igfaces, unfolded, e1_r3, e2_r3, normal_r3, e1_ig, e2_ig)

    return unfolded


def calc_stiffness(unfolded, thickness, E_array, nu):
    """
    Calculate the stiffness matrix for each of the unfolded faces using 2D plane stress formulation.
    This function assumes linear triangular elements, where the stiffness matrix for each face is computed as:
        K = thickness * A * (B^T * D * B)
    with:
      - B: strain-displacement matrix,
      - D: constitutive matrix for plane stress.

    Args:
        unfolded (np.ndarray): (nfaces, 3, 2) array representing unfolded 2D faces.
        thickness (float or np.ndarray): Thickness for each face.
        E_array (np.ndarray): (nfaces,) array of Young's moduli.
        nu (float): Poisson's ratio.

    Returns:
        np.ndarray: (nfaces, 6, 6) array representing the stiffness matrix for each face.
    """
    nfaces = unfolded.shape[0]

    # Ensure thickness is an array
    thickness = np.atleast_1d(thickness)

    # Extract x and y coordinates (shape: (nfaces, 3))
    x = unfolded[:, :, 0]
    y = unfolded[:, :, 1]

    # Compute the area of each triangle face: A = 0.5 * |(x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)|
    A = 0.5 * np.abs((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))

    # Compute coefficients for the shape function derivatives
    b1 = y[:, 1] - y[:, 2]
    b2 = y[:, 2] - y[:, 0]
    b3 = y[:, 0] - y[:, 1]
    c1 = x[:, 2] - x[:, 1]
    c2 = x[:, 0] - x[:, 2]
    c3 = x[:, 1] - x[:, 0]

    # Initialize the strain-displacement matrix B for all faces (shape: (nfaces, 3, 6))
    B = np.empty((nfaces, 3, 6))

    # Populate the B matrix using the standard form for a linear triangle
    # First row: b coefficients (for ε_xx)
    B[:, 0, 0] = b1
    B[:, 0, 2] = b2
    B[:, 0, 4] = b3
    B[:, 0, 1] = 0.0
    B[:, 0, 3] = 0.0
    B[:, 0, 5] = 0.0

    # Second row: c coefficients (for ε_yy)
    B[:, 1, 1] = c1
    B[:, 1, 3] = c2
    B[:, 1, 5] = c3
    B[:, 1, 0] = 0.0
    B[:, 1, 2] = 0.0
    B[:, 1, 4] = 0.0

    # Third row: combination for shear strain (ε_xy)
    B[:, 2, 0] = c1
    B[:, 2, 1] = b1
    B[:, 2, 2] = c2
    B[:, 2, 3] = b2
    B[:, 2, 4] = c3
    B[:, 2, 5] = b3

    # Divide each B by 2A (note: A is (nfaces,), so we reshape for broadcasting)
    B /= (2 * A)[:, None, None]

    # Construct the constitutive matrix D for plane stress for each face.
    # D = (E/(1-nu^2)) * [[1, nu, 0],
    #                     [nu, 1, 0],
    #                     [0,  0, (1-nu)/2]]
    D_const = np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    D = (E_array / (1 - nu**2))[:, None, None] * D_const  # shape: (nfaces, 3, 3)

    # Compute the stiffness matrix for each face:
    # K = thickness * A * (B^T * D * B)
    # B has shape (nfaces, 3, 6) so its transpose is (nfaces, 6, 3)
    # Using batch matrix multiplication:
    BT = np.transpose(B, (0, 2, 1))  # shape: (nfaces, 6, 3)
    # Compute D*B for each face (resulting shape: (nfaces, 3, 6))
    DB = D @ B
    # Finally, compute K for each face (shape: (nfaces, 6, 6))
    K = (thickness * A)[:, None, None] * (BT @ DB)

    return K


def assemble_Kmatrices_Fvectors(K_matrices, F_vectors, faces):
    """
    Assemble the global stiffness matrix and force vector from the individual face stiffness matrices and force vectors.
    This function uses vectorized operations and sparse matrix assembly.

    Args:
        K_matrices (np.ndarray): (nfaces, 6, 6) array of stiffness matrices for each face.
        F_vectors (np.ndarray): (nfaces, 6, 1) array of force vectors for each face.
        faces (np.ndarray): (nfaces, 3) array of vertex indices for each face.

    Returns:
        coo_matrix: Global stiffness matrix (K) in sparse COO format.
        np.ndarray: Global force vector (F).
    """
    nfaces = faces.shape[0]
    nverts = faces.max() + 1

    # Map each face's vertex indices to global DOFs:
    # For a face with vertices [v0, v1, v2], the DOFs are:
    # [2*v0, 2*v0+1, 2*v1, 2*v1+1, 2*v2, 2*v2+1]
    global_dofs = np.empty((nfaces, 6), dtype=int)
    global_dofs[:, 0] = faces[:, 0] * 2
    global_dofs[:, 1] = faces[:, 0] * 2 + 1
    global_dofs[:, 2] = faces[:, 1] * 2
    global_dofs[:, 3] = faces[:, 1] * 2 + 1
    global_dofs[:, 4] = faces[:, 2] * 2
    global_dofs[:, 5] = faces[:, 2] * 2 + 1

    # --- Assemble the Global Stiffness Matrix ---
    # For each face, create a 6x6 block of row and column indices.
    # Using np.repeat and np.tile ensures we get arrays of shape (nfaces, 36)
    rows = np.repeat(global_dofs, 6, axis=1)  # Repeat each DOF 6 times -> shape: (nfaces, 36)
    cols = np.tile(global_dofs, (1, 6))  # Tile the DOFs 6 times -> shape: (nfaces, 36)

    # Flatten to 1D arrays of length nfaces*36
    I = rows.ravel()
    J = cols.ravel()

    # Flatten the stiffness matrices: each is 6x6, so overall shape (nfaces*36,)
    data = K_matrices.reshape(-1)

    # Build the sparse global stiffness matrix.
    K_global = coo_matrix((data, (I, J)), shape=(2 * nverts, 2 * nverts))

    # --- Assemble the Global Force Vector ---
    F_global = np.zeros(2 * nverts)
    F_local = F_vectors.squeeze(axis=2)  # shape: (nfaces, 6)
    # Use np.add.at to sum the contributions for each global DOF.
    np.add.at(F_global, global_dofs, F_local)

    return K_global, F_global


def apply_pinned_roller_BC(K, F, verts2D, fix_all_dofs=False):
    """
    Applies pinned and roller boundary conditions to the global stiffness matrix and force vector.

    The node with the minimum x-coordinate is considered the pinned node (fully fixed),
    and the node with the minimum y-coordinate is considered the roller node (roller support,
    where only the y displacement is fixed).

    For each fixed degree of freedom:
      - Zero out the corresponding row and column in the stiffness matrix,
      - Set the diagonal entry to 1 (to enforce u=0 exactly),
      - Zero out the force contribution.

    The stiffness matrix is first converted to LIL format for easy modifications,
    and then converted back to CSR format.

    Args:
        K (coo_matrix): Global stiffness matrix in COO format.
        F (np.ndarray): Global force vector.
        verts2D (np.ndarray): (nverts, 2) array of 2D vertices.

    Returns:
        csr_matrix: Modified global stiffness matrix.
        np.ndarray: Modified global force vector.
    """

    if fix_all_dofs:
        # Directly return identity matrix and zero vector
        K_mod = identity(F.shape[0], format="csr", dtype=K.dtype)
        F.fill(0.0)
        return K_mod, F

    # Identify the fixed nodes:
    # Pinned node: node with minimum x-coordinate (fully fixed: x and y)
    pinned_node = np.argmin(verts2D[:, 0])
    # Roller node: node with maximum x-coordinate (only y is fixed)
    roller_node = np.argmax(verts2D[:, 0])

    # Fixed DOFs for the pinned node (both x and y directions)
    fixed_dofs = [2 * pinned_node, 2 * pinned_node + 1]

    # For the roller node, fix only the y displacement (assuming ordering: x, y)
    roller_dof = 2 * roller_node + 1
    if roller_dof not in fixed_dofs:
        fixed_dofs.append(roller_dof)

    # Convert the global stiffness matrix to LIL format for modification
    K_lil = K.tolil()

    # Apply BCs: For each fixed DOF, zero out the row and column, and set diagonal to 1
    for dof in fixed_dofs:
        K_lil[dof, :] = 0
        K_lil[:, dof] = 0
        K_lil[dof, dof] = 1

        # Also, zero out the corresponding force entry
        F[dof] = 0.0

    # Convert the modified stiffness matrix back to CSR format for efficient solving
    K_mod = K_lil.tocsr()

    return K_mod, F


def calculate_stretches_update_E_thickness(
    undeformed,
    deformed,
    initial_thickness,
    material_model="Ogden",
    nu=0.49,
    mu1=0.029159,
    alpha1=1.708,
    mu2=2.147e-06,
    alpha2=40.087,
    # Hollman parameters (add defaults or make required based on model_type)
    k1=999.91,
    n1=8.80,
    k2=0.098,
    n2=1.04,
):
    """
    Updated vectorized computation based on MATLAB code.
    For each element, the undeformed and deformed nodal coordinates are assumed
    to be stored in arrays of shape (n, 3, 2) where n is the number of elements.

    This function computes the deformation gradient F using:
      F = I + [u₁–u₃  u₂–u₃; v₁–v₃  v₂–v₃] * inv([X₁–X₃  X₂–X₃; Y₁–Y₃  Y₂–Y₃])
    then forms the right Cauchy–Green tensor C = Fᵀ F and obtains the principal stretches
    (sqrt of eigenvalues of C). These are then used (with an incompressibility assumption)
    to compute logarithmic strains, an Ogden-type hyperelastic energy, principal stresses,
    a tangent modulus vE, and an updated thickness.

    Parameters:
      undeformed       : (n,3,2) array of reference nodal coordinates (each row: [x, y])
      deformed         : (n,3,2) array of deformed nodal coordinates (each row: [x, y])
      initial_thickness: scalar, initial thickness of all elements
      nu               : Poisson's ratio (default 0.49)
      mu1, alpha1      : Ogden parameters (first term)
      mu2, alpha2      : Ogden parameters (second term)

    Returns:
      vE               : (n,) array of updated tangent moduli
      thickness_updated: (n,) array of updated thickness values
      strains_stresses : (n,5) array with columns [ε₁, ε₂, ε₃, ε̄, σ̄]

    Example usage:
    undeformed = np.array([[[0, 0], [1, 0], [0, 1]]])
    R = [[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]]
    scale = np.array([[2],[1]])
    deformed = (scale + R @ (scale * (undeformed[0].T))).T
    deformed = deformed[np.newaxis, :, :]

    initial_thickness = np.full((1,), 2.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=undeformed[0, [0,1,2,0], 0], y=undeformed[0, [0,1,2,0], 1], mode="lines"))
    fig.add_trace(go.Scatter(x=deformed[0, [0,1,2,0], 0], y=deformed[0, [0,1,2,0], 1], mode="lines"))
    fig.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1))
    fig.show()
    calculate_stretches_update_E_thickness(undeformed, deformed, initial_thickness)
    """
    # === 1. Extract the undeformed and deformed nodal coordinates ===
    # Both arrays are assumed to have shape (n,3,2)
    X_coordinates = undeformed  # (n,3,2)
    x_coordinates = deformed  # (n,3,2)

    # For each element, separate the x- and y-coordinates.
    # X and Y here correspond to the nodal coordinates of the undeformed configuration.
    X = X_coordinates[:, :, 0]  # shape: (n,3)
    Y = X_coordinates[:, :, 1]  # shape: (n,3)

    # === 2. Compute the deformation (displacement) at each node ===
    Deformation = x_coordinates - X_coordinates  # shape: (n,3,2)
    u = Deformation[:, :, 0]  # x-displacements, shape: (n,3)
    v = Deformation[:, :, 1]  # y-displacements, shape: (n,3)

    # === 3. Build the 2x2 matrices for each element ===
    # MATLAB uses node3 as a reference, so we compute differences relative to node 3.
    # For the displacement differences:
    #   [u₁-u₃,  u₂-u₃;
    #    v₁-v₃,  v₂-v₃]
    A = np.stack(
        [
            np.stack([u[:, 0] - u[:, 2], u[:, 1] - u[:, 2]], axis=1),
            np.stack([v[:, 0] - v[:, 2], v[:, 1] - v[:, 2]], axis=1),
        ],
        axis=1,
    )  # shape: (n,2,2)

    # For the undeformed coordinate differences:
    #   [X₁-X₃,  X₂-X₃;
    #    Y₁-Y₃,  Y₂-Y₃]
    B = np.stack(
        [
            np.stack([X[:, 0] - X[:, 2], X[:, 1] - X[:, 2]], axis=1),
            np.stack([Y[:, 0] - Y[:, 2], Y[:, 1] - Y[:, 2]], axis=1),
        ],
        axis=1,
    )  # shape: (n,2,2)

    # === 4. Compute the deformation gradient F ===
    # F = I + A @ inv(B) for each element (vectorized over n)
    B_inv = np.linalg.inv(B)  # shape: (n,2,2)
    F = np.eye(2)[None, :, :] + np.matmul(A, B_inv)  # shape: (n,2,2)

    # === 5. Compute the right Cauchy–Green tensor, C = Fᵀ F ===
    C = np.matmul(F.transpose(0, 2, 1), F)  # shape: (n,2,2)

    # === 6. Calculate the principal stretches ===
    # The eigenvalues of C are the squares of the principal stretches.
    eigvals = np.linalg.eigvals(C)  # shape: (n,2)
    principal_stretches = np.sqrt(np.abs(eigvals))  # shape: (n,2)
    # Ensure the first principal stretch is the larger one, second is the smaller
    principal_stretches.sort(axis=1)
    principal_stretches = principal_stretches[:, ::-1]

    # === 7. Complete the kinematics and constitutive update ===
    # Here we assume that the two computed principal stretches are λ₁ and λ₂.
    # Enforce (plane) incompressibility by setting the out-of-plane stretch to
    #   λ₃ = 1 / (λ₁ * λ₂)
    lambda1 = principal_stretches[:, 0]
    lambda2 = principal_stretches[:, 1]
    lambda3 = 1.0 / (lambda1 * lambda2)

    # Logarithmic (true) strains:
    eps1 = np.log(lambda1)
    eps2 = np.log(lambda2)
    eps3 = -(eps1 + eps2)

    # Ogden hyperelastic energy density:
    hyper_energy = (
        2 * mu2 * (lambda1**alpha2 + lambda2**alpha2 + lambda3**alpha2 - 3) / alpha2**2
        + 2 * mu1 * (lambda1**alpha1 + lambda2**alpha1 + lambda3**alpha1 - 3) / alpha1**2
    )
    hyper_energy = np.where(hyper_energy < 1e-9, 1e-9, hyper_energy)

    # Tangent modulus
    denom = eps1**2 + eps2**2 + eps3**2
    denom = np.where(denom < 1e-9, 1e-9, denom)  # to avoid division by zero

    if material_model.lower() == "ogden":
        E = 2 * (1 + nu) * hyper_energy / denom
    elif material_model.lower() == "hollman":
        eps_bar_hollman = np.sqrt(2 / 3) * np.sqrt(denom)
        sigma_bar = k1 * eps_bar_hollman**n1 + k2 * eps_bar_hollman**n2
        E = sigma_bar / eps_bar_hollman  # Denominator already has epsilon added

    # Update thickness using the out-of-plane stretch
    thickness_updated = initial_thickness * lambda3

    # Compute principal stresses (using the same ordering as for stretches)
    sigma11 = 2 * mu2 * (lambda1**alpha2 - lambda3**alpha2) / alpha2 + 2 * mu1 * (lambda1**alpha1 - lambda3**alpha1) / alpha1
    sigma22 = 2 * mu2 * (lambda2**alpha2 - lambda3**alpha2) / alpha2 + 2 * mu1 * (lambda2**alpha1 - lambda3**alpha1) / alpha1
    sigma_bar = np.sqrt(sigma11**2 + sigma22**2 - sigma11 * sigma22)
    eps_bar = np.sqrt(eps1**2 + eps2**2 + eps3**2 - eps1 * eps2 - eps1 * eps3 - eps2 * eps3)

    # Assemble strain and stress measures: [ε₁, ε₂, ε₃, ε̄, σ̄]
    strains_stresses = np.column_stack((eps1, eps2, eps3, eps_bar, sigma_bar))

    return E, thickness_updated, strains_stresses


def add_uv_coords(surf_mesh: pv.PolyData, plane_mesh: pv.PolyData):
    # 1. Prep Meshes & Calculate Shared Normalized UVs (Aspect Preserving)
    surf_mesh_out = surf_mesh.copy(deep=True)
    plane_mesh_out = plane_mesh.copy(deep=True)  # Create copy for UV mesh too
    uvs = plane_mesh.points[:, :2].copy()  # Source UVs from plane_mesh points

    min_uv, max_uv = uvs.min(0), uvs.max(0)
    range_uv = max_uv - min_uv
    range_uv[range_uv < 1e-9] = 0.0
    max_r = max(range_uv) if max(range_uv) > 1e-9 else 1.0
    if max(range_uv) < 1e-9:
        norm_uvs = np.full_like(uvs, 0.5)
    else:
        norm_uvs = ((uvs - min_uv) / max_r) + ((1.0 - (range_uv / max_r)) / 2.0)
    norm_uvs = np.clip(norm_uvs, 0.0, 1.0)  # Ensure bounds

    # Assign SAME normalized UVs to BOTH output meshes
    surf_mesh_out.active_texture_coordinates = norm_uvs
    plane_mesh_out.active_texture_coordinates = norm_uvs  # Apply here too

    return surf_mesh_out, plane_mesh_out


def plot_hist_box(data, filename, x_range=None, y_range=None, title=None, hist_bin_size=None):
    fig = go.Figure(
        [
            go.Histogram(
                x=data,
                xbins=dict(size=hist_bin_size) if hist_bin_size else None,
                marker_color="#1f77b4",
                name="Histogram",
                histnorm="percent",
                yaxis="y1",
            ),
            go.Box(
                x=data,
                yaxis="y2",
                orientation="h",
                boxmean="sd",
                marker_color="orange",
                fillcolor="orange",
                opacity=0.7,
                line=dict(color="black", width=3),
                name="Box Plot",
                boxpoints=False,
            ),
        ]
    )
    fig.update_layout(
        title=title or "",
        xaxis=dict(range=x_range, zeroline=False),
        yaxis=dict(range=y_range, title="Percentage", anchor="x"),
        yaxis2=dict(showticklabels=False, anchor="x", domain=[0.2, 1]),
        bargap=0.1,
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=60, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.write_html(Path(f"outputs/{time.strftime('%H%M%S')}.{int((time.time() % 1)*1000):03d}_{filename}"))


def hiFEM(
    verts3D,
    faces,
    thickness=2.0,
    nu=0.49,
    material_model="Ogden",
    ig_verts=None,  # Initial guess mesh (optional, if none, BFF will be used to create one)
    fix_all_dofs=False,  # For debugging.
    max_iters=10,
    convergence_tol=1e-5,
    verbose=False,
    scene_obj=None,
):
    """hiFEM implementation with vectorized computations.
    To read more about the core principle and more details, please read the paper and hifem.md

    Args:
        verts3D (np.ndarray): (n,3) array of 3D vertex coordinates.
        faces (np.ndarray): (m,3) array of triangle indices.
    """
    if ig_verts is None:
        # Create the initial guess for the 2D layout using BFF
        cm = BFF(verts3D, faces)  # CETM also works near identical. Use like: cm = CETM(verts3D, faces)
        image = cm.layout()
        uv_verts = image.vertices
        verts2D = match_surface_area(verts3D, faces, uv_verts)
    else:
        # Use the provided initial guess mesh
        verts2D = ig_verts[:, :2]

    igfaces = verts2D[faces]  # the initial guess faces.
    r3faces = verts3D[faces]  # the reference 3D faces.
    unfolded = unfold_faces(r3faces, igfaces)

    initial_thickness = np.full((faces.shape[0],), thickness)
    thickness_updated = initial_thickness
    E_array = np.full((faces.shape[0],), 1.0)

    # --- Main inverse FEM loop ---
    # ------------------------------
    E_steps = [E_array]
    L2norm_steps = []
    for i in range(max_iters):
        K_matrices = calc_stiffness(unfolded, thickness_updated, E_array, nu)
        U_vectors = unfolded - igfaces
        F_vectors = K_matrices @ U_vectors.reshape(-1, 6, 1)
        K, F = assemble_Kmatrices_Fvectors(K_matrices, F_vectors, faces)

        K_mod, F_mod = apply_pinned_roller_BC(K, F, verts2D, fix_all_dofs=fix_all_dofs)

        # Solve the system K_mod * U = F_mod
        disps = spsolve(K_mod, F_mod)
        U_disp = disps.reshape(verts2D.shape[0], 2)

        updated = verts2D + U_disp

        E_array, thickness_updated, strains_stresses = calculate_stretches_update_E_thickness(
            undeformed=updated[faces],
            deformed=unfolded,
            initial_thickness=initial_thickness,
            material_model=material_model,
        )
        E_steps.append(E_array)

        # Convergence check
        V2 = E_steps[-1]
        V1 = E_steps[-2]
        L2norm = np.linalg.norm((V2 - V1) / V1)
        print(f"Step {i}: L2norm = {L2norm}")
        L2norm_steps.append(L2norm)
        if L2norm < convergence_tol:
            break

    # Save the convergance history
    os.makedirs(Path("outputs"), exist_ok=True)
    np.savetxt(Path(f"outputs/{time.strftime('%H%M%S')}.{int((time.time() % 1) * 1000):03d}_L2norm.txt"), np.array(L2norm_steps))

    def rotate_to_x_axis(nodes_2d):
        """
        A Helper function: Aligns 2D points so their first principal direction aligns with the X-axis using SVD.

        Args:
            nodes_2d (np.ndarray): Array of shape (N, 2) with (x, y) coordinates.

        Returns:
            np.ndarray: Array of shape (N, 2) with aligned (x, y) coordinates.
        """
        centroid = np.mean(nodes_2d, axis=0)
        centered_nodes = nodes_2d - centroid
        _, _, Vh = np.linalg.svd(centered_nodes, full_matrices=False)
        angle_rad = np.arctan2(Vh[0, 1], Vh[0, 0])  # Angle of 1st principal component
        cos_a, sin_a = np.cos(-angle_rad), np.sin(-angle_rad)
        # Rotation matrix transpose (for applying to rows of points)
        rot_matrix_T = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        aligned_nodes = centered_nodes @ rot_matrix_T + centroid
        return aligned_nodes

    # The updated 2D coordinates are now in `updated`
    updated = rotate_to_x_axis(updated)
    plane_mesh = pv.PolyData(
        np.hstack([updated, np.zeros((updated.shape[0], 1))]),
        np.hstack([np.full((len(faces), 1), 3), faces]),
    )
    surf_mesh = pv.PolyData(verts3D, np.hstack([np.full((len(faces), 1), 3), faces]))
    surf_mesh, plane_mesh = add_uv_coords(surf_mesh, plane_mesh)

    # Align the surfaces
    verts_aligned = align_surfaces(updated, verts3D, faces, offset=4.0)
    aligned_mesh = pv.PolyData(verts_aligned, np.hstack([np.full((len(faces), 1), 3), faces]))
    aligned_mesh, plane_mesh = add_uv_coords(aligned_mesh, plane_mesh)

    if verbose:
        # --- Save the surf_mesh and plane_mesh and aligned ---
        # -----------------------------------------------------
        texture = pv.read_texture(Path("src/chess.jpg"))
        light = pv.Light(position=(0, 0, 10), intensity=0.2, light_type="scene light")  # Try 'headlight' or 'camera light' too
        plotter = pv.Plotter(off_screen=True)
        plotter.add_light(light)
        plotter.add_mesh(surf_mesh, texture=texture, smooth_shading=True, show_edges=False, color=None)
        if scene_obj:
            plotter.add_mesh(scene_obj, color="red", show_edges=False, smooth_shading=False)
        plotter.export_html(Path(f"outputs/{time.strftime('%H%M%S')}.{int((time.time() % 1) * 1000):03d}_surf_mesh.html"))

        plotter = pv.Plotter(notebook=False)
        plotter.add_light(light)
        plotter.add_mesh(pv.Arrow(start=[0, 0, 0], direction=[1, 0, 0], scale=5), color="red", label="X-axis")
        plotter.add_mesh(pv.Arrow(start=[0, 0, 0], direction=[0, 1, 0], scale=5), color="green", label="Y-axis")
        plotter.add_mesh(pv.Arrow(start=[0, 0, 0], direction=[0, 0, 1], scale=5), color="blue", label="Z-axis")
        plotter.add_mesh(plane_mesh, texture=texture, smooth_shading=True, show_edges=False, color=None)
        plotter.camera_position = "xy"
        plotter.export_html(Path(f"outputs/{time.strftime('%H%M%S')}.{int((time.time() % 1) * 1000):03d}_plane_mesh.html"))

        plotter = pv.Plotter(off_screen=True)
        plotter.add_light(light)
        plotter.add_mesh(surf_mesh, texture=texture, smooth_shading=True, show_edges=False, color=None)
        plotter.add_mesh(aligned_mesh, texture=texture, smooth_shading=True, show_edges=False, color=None)
        if scene_obj:
            plotter.add_mesh(scene_obj, color="red", show_edges=False, smooth_shading=False)
        plotter.export_html(Path(f"outputs/{time.strftime('%H%M%S')}.{int((time.time() % 1) * 1000):03d}_flap_aligned_textured.html"))

        # --- 1st Principal Stretch ---
        # -----------------------------
        plot_hist_box(
            np.exp(strains_stresses[:, 0]),
            x_range=[0.8, 1.3],
            y_range=[0, 6],
            hist_bin_size=0.005,
            title="First Principal Stretch Histogram",
            filename="deformation_1st_principal_stretch_histogram.html",
        )

        plane_mesh.cell_data["First Principal Stretch"] = np.exp(strains_stresses[:, 0])
        plotter = pv.Plotter(notebook=False)
        plotter.add_light(light)
        plotter.add_mesh(plane_mesh, scalars="First Principal Stretch", cmap="jet", show_edges=False)
        plotter.add_scalar_bar(title="First Principal Stretch")
        plotter.add_title("First Principal Stretch")
        plotter.camera_position = "xy"
        plotter.export_html(Path(f"outputs/{time.strftime('%H%M%S')}.{int((time.time() % 1) * 1000):03d}_deformation_1st_principal_stretch.html"))

        # --- 2nd Principal Stretch ---
        # -----------------------------
        plot_hist_box(
            np.exp(strains_stresses[:, 1]),
            x_range=[0.8, 1.3],
            y_range=[0, 6],
            hist_bin_size=0.005,
            title="Second Principal Stretch Histogram",
            filename="deformation_2nd_principal_stretch_histogram.html",
        )

        plane_mesh.cell_data["Second Principal Stretch"] = np.exp(strains_stresses[:, 1])
        plotter = pv.Plotter(notebook=False)
        plotter.add_light(light)
        plotter.add_mesh(plane_mesh, scalars="Second Principal Stretch", cmap="jet", show_edges=False)
        plotter.add_scalar_bar(title="Second Principal Stretch")
        plotter.add_title("Second Principal Stretch")
        plotter.camera_position = "xy"
        plotter.export_html(Path(f"outputs/{time.strftime('%H%M%S')}.{int((time.time() % 1) * 1000):03d}_deformation_2nd_principal_stretch.html"))

        # --- Area Stretch ---
        # --------------------
        plot_hist_box(
            np.exp(strains_stresses[:, 0]) * np.exp(strains_stresses[:, 1]),
            x_range=[0.7, 1.6],
            y_range=[0, 6],
            hist_bin_size=0.005,
            title="Area Stretch Histogram",
            filename="deformation_area_stretch_histogram.html",
        )

        plane_mesh.cell_data["Area Stretch"] = np.exp(strains_stresses[:, 0]) * np.exp(strains_stresses[:, 1])
        plotter = pv.Plotter(notebook=False)
        plotter.add_light(light)
        plotter.add_mesh(plane_mesh, scalars="Area Stretch", cmap="jet", show_edges=False)
        plotter.add_scalar_bar(title="Area Stretch")
        plotter.add_title("Area Stretch")
        plotter.camera_position = "xy"
        plotter.export_html(Path(f"outputs/{time.strftime('%H%M%S')}.{int((time.time() % 1) * 1000):03d}_deformation_area_stretch.html"))

        # -- Area Shear --
        # ------------------
        # plot_hist_box(
        #     np.exp(strains_stresses[:, 0]) / np.exp(strains_stresses[:, 1]),
        #     x_range=[0.9, 2],
        #     y_range=[0, 20],
        #     hist_bin_size=0.1,
        #     title="Area Shear Histogram",
        #     filename="deformation_area_shear_histogram.html",
        # )

        # plane_mesh.cell_data["Area Shear"] = np.exp(strains_stresses[:, 0]) / np.exp(strains_stresses[:, 1])
        # plotter = pv.Plotter(notebook=False)
        # plotter.add_light(light)
        # plotter.add_mesh(plane_mesh, scalars="Area Shear", cmap="jet", show_edges=False)
        # plotter.add_scalar_bar(title="Area Shear")
        # plotter.add_title("Area Shear")
        # plotter.camera_position = "xy"
        # plotter.export_html(Path(f"outputs/{time.strftime('%H%M%S')}.{int((time.time() % 1) * 1000):03d}_deformation_area_shear.html"))

        # Effective strains on plane_mesh
        plane_mesh.cell_data["Effective Strain"] = strains_stresses[:, -2]
        plotter = pv.Plotter(notebook=False)
        plotter.add_light(light)
        plotter.add_mesh(plane_mesh, scalars="Effective Strain", cmap="jet", show_edges=False)
        plotter.add_scalar_bar(title="Effective Strain")
        plotter.add_title("Effective Strain")
        plotter.camera_position = "xy"
        plotter.export_html(Path(f"outputs/{time.strftime('%H%M%S')}.{int((time.time() % 1) * 1000):03d}_deformation_effective_strain.html"))

        # Effective strains on surf_mesh
        surf_mesh.cell_data["Effective Strain"] = strains_stresses[:, -2]
        plotter = pv.Plotter(notebook=False)
        plotter.add_light(light)
        plotter.add_mesh(surf_mesh, scalars="Effective Strain", cmap="jet", show_edges=False)
        plotter.add_scalar_bar(title="Effective Strain")
        plotter.add_title("Effective Strain")
        if scene_obj:
            plotter.add_mesh(scene_obj, color="white", show_edges=False, smooth_shading=False)
        plotter.export_html(Path(f"outputs/{time.strftime('%H%M%S')}.{int((time.time() % 1) * 1000):03d}_deformation_effective_strain3D.html"))

        # Effective stress on plane_mesh
        plane_mesh.cell_data["Effective Stress"] = strains_stresses[:, -1]
        plotter = pv.Plotter(notebook=False)
        plotter.add_light(light)
        plotter.add_mesh(plane_mesh, scalars="Effective Stress", cmap="jet", show_edges=False)
        plotter.add_scalar_bar(title="Effective Stress")
        plotter.add_title("Effective Stress")
        plotter.camera_position = "xy"
        plotter.export_html(Path(f"outputs/{time.strftime('%H%M%S')}.{int((time.time() % 1) * 1000):03d}_deformation_effective_stress.html"))

        # Effective stress on surf_mesh
        surf_mesh.cell_data["Effective Stress"] = strains_stresses[:, -1]
        plotter = pv.Plotter(notebook=False)
        plotter.add_light(light)
        plotter.add_mesh(surf_mesh, scalars="Effective Stress", cmap="jet", show_edges=False)
        plotter.add_scalar_bar(title="Effective Stress")
        plotter.add_title("Effective Stress")
        if scene_obj:
            plotter.add_mesh(scene_obj, color="white", show_edges=False, smooth_shading=False)
        plotter.export_html(Path(f"outputs/{time.strftime('%H%M%S')}.{int((time.time() % 1) * 1000):03d}_deformation_effective_stress3D.html"))

        # # Tangent modulus on plane_mesh
        # plane_mesh.cell_data["Tangent Modulus"] = strains_stresses[:, -1]
        # plotter = pv.Plotter(notebook=False)
        # plotter.add_light(light)
        # plotter.add_mesh(plane_mesh, scalars="Tangent Modulus", cmap="jet", show_edges=False)
        # plotter.add_scalar_bar(title="Tangent Modulus")
        # plotter.add_title("Tangent Modulus")
        # plotter.camera_position = "xy"
        # plotter.export_html(Path(f"outputs/{time.strftime('%H%M%S')}.{int((time.time() % 1) * 1000):03d}_deformation_tangent_modulus.html"))

        # Thickness on plane_mesh
        plane_mesh.cell_data["Thickness"] = thickness_updated
        plotter = pv.Plotter(notebook=False)
        plotter.add_light(light)
        plotter.add_mesh(plane_mesh, scalars="Thickness", cmap="jet", show_edges=False)
        plotter.add_scalar_bar(title="Thickness")
        plotter.add_title("Thickness")
        plotter.camera_position = "xy"
        plotter.export_html(Path(f"outputs/{time.strftime('%H%M%S')}.{int((time.time() % 1) * 1000):03d}_deformation_thickness.html"))

    return plane_mesh, aligned_mesh


if __name__ == "__main__":
    ### Test faces:
    # verts2D = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    # verts3D = np.array([[0., 0., 0.], [1., 0., 1.], [0., 1., 1.]])
    # faces = np.array([[0, 1, 2]])

    surf3D_mesh = pv.read("inputs/flap3D.obj")
    verts3D = surf3D_mesh.points
    faces = surf3D_mesh.faces.reshape(-1, 4)[:, 1:]
    hiFEM(verts3D, faces)
