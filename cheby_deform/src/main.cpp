#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuizmoWidget.h>
#include <igl/opengl/glfw/imgui/SelectionWidget.h>
#include <GLFW/glfw3.h>
#include <igl/unproject_onto_mesh.h>
#include<igl/Timer.h>
#include<igl/cotmatrix.h>
#include<igl/massmatrix.h>
#include<igl/invert_diag.h>
#include <igl/cotmatrix_entries.h>
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <igl/floor.h>
#include <igl/screen_space_selection.h>
#include <igl/adjacency_list.h>
#include <igl/readPLY.h>
#include <igl/screen_space_selection.h>
#include <igl/AABB.h>
#include <igl/local_basis.h>
#include <igl/grad.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/lscm.h>
#include <igl/adjacency_matrix.h>
#include <igl/sum.h>
#include <igl/speye.h>
#include <igl/repdiag.h>
#include <igl/cat.h>
#include <igl/dijkstra.h>
#include <igl/exact_geodesic.h>
#include <igl/writeOFF.h>
#include <igl/doublearea.h>
#include <igl/rotate_vectors.h>
#include <igl/signed_distance.h>
#include <igl/ray_mesh_intersect.h>
#include <igl/per_face_normals.h>
#include <igl/writeOBJ.h>
#include <igl/vertex_components.h>
#include <igl/stb/read_image.h>

//names
using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

//mesh stuff
Eigen::MatrixXd V_orig, V_def, V;//vertex matrices
Eigen::MatrixXi F;//face matrix
Eigen::SparseMatrix<double> L, M, M_tr, M_inv, Mass, Du, Dv;//laplacian matrix, mass matrix
VectorXd triangle_area;
Eigen::MatrixXd Cov;//for rotation fitting
std::vector<std::set<int>> areas;//each one contains itself and all others attached
Eigen::SimplicialCholesky<SparseMatrix<double>> solver;
Eigen::SparseMatrix<double> free_influenced, J;
double lambda = 0.0;
std::vector<std::vector<int>> adj_list;
Eigen::VectorXi cvi;//save how many triangles vertices are part of
struct Edge {
    const int i;
    const int j;
    const int rid;
    const double w;
};
std::vector<std::vector<Edge>> edgeSets;
std::vector<std::vector<Edge>> edgeSets_tr;
double dq_norm;
Eigen::MatrixXd VN;//vertex normals
Eigen::MatrixXd FN_collision;//face normals
Eigen::MatrixXd UV;// UV coordinates, #V x2
bool showingUV = false;
bool freeBoundary = true;
double TextureResolution = 1;
igl::opengl::ViewerCore temp3D;
igl::opengl::ViewerCore temp2D;
double max_dist = 0;
int v1, v2;
bool cheby = false;
bool print = false;
VectorXd per_face_energy;
double global_energy;
double actual_energy;
bool should_decrease = false;
bool show_texture = true;
double PI = 3.141592;

//UI deformation stuff
Viewer viewer;
bool vertex_picking_mode = false;
bool handle_deleting_mode = false;
//list of all vertices with their corresponding handle id. -1 if no handle
Eigen::VectorXi handle_id(0, 1);
//list of all vertices belonging to handles (id not -1), #HV x1
Eigen::VectorXi handle_vertices(0, 1);
//updated positions of handle vertices, #HV x3
Eigen::MatrixXd handle_vertex_positions(0, 3);
int num_handles = 0;
igl::opengl::glfw::imgui::ImGuizmoWidget guizmo;
MatrixXd pluginpos;
Eigen::Matrix4f T0 = guizmo.T;
igl::opengl::glfw::imgui::SelectionWidget selection;
Eigen::Array<double, Eigen::Dynamic, 1> and_visible;
igl::opengl::glfw::imgui::ImGuiMenu menu;
int plugin_vertex = 0;
Eigen::VectorXi v_free_index, v_constrained_index;
VectorXd sel_vertices;
//menu option stuff
enum Handle { LASSO, MARQUE, VERTEX, REMOVE, NONE };
Handle handle_option = NONE;
enum Trans { ROTATE, TRANSLATE, SCALE };
Trans transform_mode = TRANSLATE;
enum Method { PARAMETRIZE, TRIANGLE };
Method method_mode = TRIANGLE;
int method = 1;
Eigen::RowVector3d point_color(44/ 255.0, 77 / 255.0,117 / 255.0);
Eigen::RowVector3d mesh_color(146 / 255.0, 187 / 255.0, 227 / 255.0);
MatrixXd F1, F2, F3;//local basis
std::vector<Eigen::MatrixXd> old_cheby_grad;
double limit_shearing = 90;
//parameters for dynamics
double dyn_ym = 2;
double dyn_h = 0.001;
double dyn_force = -9.81;
bool use_gravity = false;
double dyn_dw;
MatrixXd dyn_vel;
MatrixXd dyn_f_ext;
MatrixXd param_3d;
bool start_from_param = false;
bool show_net = false;
SparseMatrix<double> L_orig, L_curr, L_uv, L_pl, L_orig_flat;
MatrixXd C_tr;
double no_bend = 0;
MatrixXi F_tex;
MatrixXd V_collision_plus;
MatrixXd V_collision;
MatrixXi F_collision;
bool use_collision = false;
VectorXd collision_out_weights;
vector<bool> was_in;
vector<set<int>> v_to_uv_corr;
vector<int> uv_to_v_corr;
bool change_uv_not_fac = false;
vector<vector<int>> comp_to_seam_idx;
vector<vector<tuple<int, int>>> seam_vertices;//seam idx--> list of vertex correspondences (tuple has the idx of the uv vertex on both sides)
vector<vector<Vector3d>> seam_vertex_lap;
int animation_frame = 0;
Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R,G,B,A;


//parameterization
bool computed_free = false;
bool first_solve=true;
vector<Matrix2d> Vs;
vector<Matrix2d> Ss;
vector<Matrix2d> Us;

void factorize(Viewer& viewer, double lambda);

//helper functions
void Redraw()
{
    viewer.data(0).clear();

    if (!showingUV)
    {
        viewer.data(0).set_mesh(V, F);
        viewer.data(0).set_face_based(false);
        viewer.data(0).show_lines = true;
        if (UV.size() != 0)
        {
            // Read the PNG
            //igl::stb::read_image("../data/texture-silk.png",R,G,B,A);
            viewer.data(0).set_uv(TextureResolution * UV, F_tex);
            viewer.data(0).show_texture = show_texture;
            //viewer.data(0).set_texture(R,G,B);
        }
    }
    else
    {
        viewer.data(0).set_mesh(UV, F_tex);
        viewer.data(0).set_uv(TextureResolution * UV, F_tex);
        viewer.data(0).show_texture = show_texture;
        //viewer.data(0).set_texture(R,G,B);

    }
    viewer.data(0).set_colors(mesh_color);
}

static inline void SSVD2x2(const Eigen::Matrix2d& J, Eigen::Matrix2d& U, Eigen::Matrix2d& S, Eigen::Matrix2d& V)
{
    double e = (J(0) + J(3)) * 0.5;
    double f = (J(0) - J(3)) * 0.5;
    double g = (J(1) + J(2)) * 0.5;
    double h = (J(1) - J(2)) * 0.5;
    double q = sqrt((e * e) + (h * h));
    double r = sqrt((f * f) + (g * g));
    double a1 = atan2(g, f);
    double a2 = atan2(h, e);
    double rho = (a2 - a1) * 0.5;
    double phi = (a2 + a1) * 0.5;

    S(0) = q + r;
    S(1) = 0;
    S(2) = 0;
    S(3) = q - r;

    double c = cos(phi);
    double s = sin(phi);
    U(0) = c;
    U(1) = s;
    U(2) = -s;
    U(3) = c;

    c = cos(rho);
    s = sin(rho);
    V(0) = c;
    V(1) = -s;
    V(2) = s;
    V(3) = c;
}

void compute_handle_centroid(MatrixXd& handle_centroid, int id)
{
    //compute centroid of handle
    handle_centroid.setZero(1, 3);
    int num = 0;
    for (long vi = 0; vi < V.rows(); ++vi)
    {
        int r = handle_id[vi];
        if (r == id)
        {
            handle_centroid += V.row(vi);
            num++;
        }
    }
    handle_centroid = handle_centroid.array() / num;
}

bool make_area_handle() {
    vector<int> area_handle_vertices;
    set<int> contained_handles;
    for (int i = 0; i < sel_vertices.size(); i++) {
        if (sel_vertices(i) > 0.9 && and_visible(i)) {
            area_handle_vertices.push_back(i);//vertex index i is part of this handle
            if (handle_id(i) != -1) {
                contained_handles.insert(i);
            }
        }
    }
    if(area_handle_vertices.size()==0){
        return false;//empty handle, do nothing
    }
    int id = area_handle_vertices[0];
    int old_num_handles = num_handles;
    num_handles += area_handle_vertices.size() - contained_handles.size();
    Eigen::VectorXi up_handle_vertices = Eigen::VectorXi(num_handles);//update handle vertex vector
    Eigen::MatrixXd up_handle_pos = Eigen::MatrixXd(num_handles, 3);//update handle vertex vector
    int count = 0;
    for (int i = 0; i < old_num_handles; i++) {
        if (contained_handles.count(handle_vertices(i)) == 0) {
            up_handle_vertices(count) = handle_vertices(i);
            up_handle_pos.row(count) = handle_vertex_positions.row(i);
            count++;
        }
    }
    for (int i = 0; i < area_handle_vertices.size(); i++) {
        handle_id(area_handle_vertices[i]) = id;
        up_handle_vertices(count + i) = area_handle_vertices[i];
        up_handle_pos.row(count + i) = V.row(area_handle_vertices[i]);
    }
    handle_vertices = up_handle_vertices;
    handle_vertex_positions = up_handle_pos;
    plugin_vertex = area_handle_vertices[0];
    viewer.data().clear_points();
    viewer.data().add_points(handle_vertex_positions, point_color);
    compute_handle_centroid(pluginpos, id);
    guizmo.T.block(0, 3, 3, 1) = pluginpos.transpose().cast<float>();
    guizmo.visible = true;
    factorize(viewer, lambda);
    selection.mode = igl::opengl::glfw::imgui::SelectionWidget::OFF;
    handle_option = NONE;
    return true;
}

static void computeSurfaceGradientMatrix(MatrixXd V_uv, MatrixXi F_uv, MatrixXd V_msh, MatrixXi F_msh, SparseMatrix<double>& D1, SparseMatrix<double>& D2)
{
    MatrixXd F1, F2, F3;
    SparseMatrix<double> DD, Dx, Dy;

    igl::grad(V_uv, F_uv, DD);//only outputs as many gradients as dimension, so fine in 3d and 2d:)
    Dx.resize(F_msh.rows(), V_msh.rows());
    Dx.reserve(F_msh.rows() * 3);
    Dy.resize(F_msh.rows(), V_msh.rows());
    Dy.reserve(F_msh.rows() * 3);
    uv_to_v_corr.resize(V_uv.rows());
    for (int f = 0; f < F_msh.rows(); f++) {
        uv_to_v_corr[F_uv(f, 0)] = F_msh(f, 0);
        uv_to_v_corr[F_uv(f, 1)] = F_msh(f, 1);
        uv_to_v_corr[F_uv(f, 2)] = F_msh(f, 2);
        Dx.insert(f, F_msh(f, 0)) = DD.coeff(f, F_uv(f, 0));
        Dx.insert(f, F_msh(f, 1)) = DD.coeff(f, F_uv(f, 1));
        Dx.insert(f, F_msh(f, 2)) = DD.coeff(f, F_uv(f, 2));
        Dy.insert(f, F_msh(f, 0)) = DD.coeff(f + F_msh.rows(), F_uv(f, 0));
        Dy.insert(f, F_msh(f, 1)) = DD.coeff(f + F_msh.rows(), F_uv(f, 1));
        Dy.insert(f, F_msh(f, 2)) = DD.coeff(f + F_msh.rows(), F_uv(f, 2));
    }
    D1 = Dx;
    D2 = Dy;
}

//LSCM and Chebyshev deformation to initialize if texture is not given
void ConvertConstraintsToMatrixForm(VectorXi indices, MatrixXd positions, Eigen::SparseMatrix<double>& C, VectorXd& d)
{
    int con = indices.size();
    C.resize(2 * con, 2 * V.rows());
    C.reserve(2 * con);
    d.resize(2 * con);
    d.setZero();
    for (int i = 0; i < con; i++) {
        //build pseudo identity matrix (only id at constrained positions)
        C.insert(i, indices(i)) = 1;//u
        C.insert(i + con, indices(i) + V.rows()) = 1;//v
        //build positions
        d(i) = positions(i, 0);
        d(i + con) = positions(i, 1);
    }
}

static void computeSurfaceGradientMatrix_para(MatrixXd V, SparseMatrix<double>& D1, SparseMatrix<double>& D2){
    MatrixXd F1, F2, F3;
    SparseMatrix<double> DD, Dx, Dy, Dz;

    igl::local_basis(V, F, F1, F2, F3);
    igl::grad(V, F, DD);
    Dx = DD.topLeftCorner(F.rows(), V.rows());
    Dy = DD.block(F.rows(), 0, F.rows(), V.rows());
    Dz = DD.bottomRightCorner(F.rows(), V.rows());
    //change to local coordinate system
    D1 = F1.col(0).asDiagonal() * Dx + F1.col(1).asDiagonal() * Dy + F1.col(2).asDiagonal() * Dz;
    D2 = F2.col(0).asDiagonal() * Dx + F2.col(1).asDiagonal() * Dy + F2.col(2).asDiagonal() * Dz;

    // conversion in case V and UV don't match
    // Define triplet lists for Dx and Dy
	std::vector<Eigen::Triplet<double>> triplets_Dx;
	std::vector<Eigen::Triplet<double>> triplets_Dy;

	// Fill triplets
    for (int f = 0; f < F_tex.rows(); f++) {
		for (int i = 0; i < 3; ++i) {
			int tex_index = F_tex(f, i);
			int vertex_index = F(f, i);
			triplets_Dx.push_back(Eigen::Triplet<double>(f, tex_index, D1.coeff(f, vertex_index)));
			triplets_Dy.push_back(Eigen::Triplet<double>(f, tex_index, D2.coeff(f, vertex_index)));
		}
	}

	// Construct Dx and Dy matrices from triplets
	Dx.resize(F_tex.rows(), V.rows());
	Dy.resize(F_tex.rows(), V.rows());
	Dx.setFromTriplets(triplets_Dx.begin(), triplets_Dx.end());
	Dy.setFromTriplets(triplets_Dy.begin(), triplets_Dy.end());
    //final gradients to return
	D1 = Dx;
	D2 = Dy;
}

void computeParameterization(int type)
{
    VectorXi fixed_UV_indices;
    MatrixXd fixed_UV_positions;

    SparseMatrix<double> A;
    VectorXd b;
    Eigen::SparseMatrix<double> C;
    VectorXd d;
    
    //compute fixed vertex positions
    if (!computed_free) {//dont't recompute for efficiency
        MatrixXi boundary;
        igl::boundary_loop(F, boundary);
        VectorXi VS, VT;
        VectorXi FS, FT;
        FS.resize(0);
        FT.resize(0);
        VS.resize(1);
        max_dist = 0;
        VectorXd Doi;
        VS(0) = boundary(0);
        VT = boundary;
        igl::exact_geodesic(V, F, VS, FS, VT, FT, Doi);
        for (int j = 0; j < Doi.size(); j++) {
            if (Doi(j) > max_dist) {
                v1=boundary(0);
                v2 = VT(j);
                max_dist = Doi(j);
            }
        }
        computed_free = true;
    }
    if(type==1){//lscm needs two fixed
        fixed_UV_indices.resize(2);
        fixed_UV_indices(0) = v1;
        fixed_UV_indices(1) = v2;
        fixed_UV_positions.resize(2, 2);
        fixed_UV_positions << 0, 0, max_dist, 0;
    }
    else{//cheby needs only one
        fixed_UV_indices.resize(1);
        fixed_UV_indices(0) = v1;
        fixed_UV_positions.resize(1, 2);
        fixed_UV_positions << 0, 0;
    }
    ConvertConstraintsToMatrixForm(fixed_UV_indices, fixed_UV_positions, C, d);
    // system matrix
    A.resize(2 * V.rows(), 2 * V.rows());
    int con = fixed_UV_indices.size();
    if (type == 1) {//lscm
        SparseMatrix<double> Dx, Dy;
        computeSurfaceGradientMatrix_para(V_orig, Dx, Dy);//results are #Fx#V
        vector<vector<SparseMatrix<double>>> matrices;
        matrices.resize(2);
        matrices[0].push_back(Dx);
        matrices[0].push_back(-Dy);
        matrices[1].push_back(Dy);
        matrices[1].push_back(Dx);
        igl::cat(matrices, A);
        SparseMatrix<double> M, Mass;
        VectorXd area;
        igl::doublearea(V_orig, F, area);
        M = area.asDiagonal();// igl::diag(area, M);
        vector<vector<SparseMatrix<double>>> matrices2;
        matrices2.resize(2);
        SparseMatrix<double> zero;
        zero.resize(F.rows(), F.rows());
        zero.setZero();
        matrices2[0].push_back(M);
        matrices2[0].push_back(zero);
        matrices2[1].push_back(zero);
        matrices2[1].push_back(M);
        igl::cat(matrices2, Mass);
        A = (A.transpose() * Mass * A).eval();
        b.resize(V.rows() * 2);
        b.setZero();
        //set size for cheby iterations
        Vs.resize(F.rows());
		Us.resize(F.rows());
		Ss.resize(F.rows());
    }
    if (type == 2) {//chebyshev
        SparseMatrix<double> Dx, Dy, J;
        computeSurfaceGradientMatrix_para(V_orig, Dx, Dy);//results are #Fx#V
        vector<vector<SparseMatrix<double>>> mat;
        SparseMatrix<double> dzero;
        VectorXd Ju, Jv;
        dzero.resize(Dx.rows(), Dx.cols());
        dzero.setZero();
        mat.resize(2);
        mat[0].push_back(Dx);
        mat[1].push_back(Dy);
        igl::cat(mat, J);
        Ju = J * UV.col(0);
        Jv = J * UV.col(1);
        double PI = 3.141592;
        VectorXd R(4 * F.rows());

        for (int f = 0; f < F.rows(); f++) {
            Eigen::Matrix2d Juv;
            Juv << Ju(f), Ju(f + F.rows()), Jv(f), Jv(f + F.rows());
            Eigen::Matrix2d U, S, Vf;
            Matrix2d Ri;
            MatrixXd I = Matrix2d::Identity();
            Matrix2d fake_inverse;

            if(first_solve){
                fake_inverse << Juv(1, 1), -Juv(0, 1), -Juv(1, 0), Juv(0, 0);
                SSVD2x2(((fake_inverse + 0.00001 * I).colwise().normalized()).inverse(), U, S, Vf);
				Us[f] = U;
				Vs[f] = Vf;
				Ss[f] = S;
            }
            else {
				U = Us[f];
				Vf = Vs[f];
				S = Ss[f];
			}

            //based on this do sigma fitting now
			//entries to approximate (not exactly svd values)
			//a>b
			double a = (U.transpose() * Juv * Vf)(0, 0);
			double b = (U.transpose() * Juv * Vf)(1, 1);

            //first naive projection step: solve inverse case optimally
			double sig1_naive = sqrt(2 / ((a * a) / (b * b) + 1));
			double sig2_naive = sqrt(2 - sig1_naive * sig1_naive);
			if (a > 2) {//note:cutoff case for better projection
				sig1_naive = 1 / a;
				sig2_naive = sqrt(2 - sig1_naive * sig1_naive);
			}
			sig1_naive = 1 / sig1_naive;
			sig2_naive = 1 / sig2_naive;

			// approximate around naive guess (Taylor's version) to refine guess
			double c1 = sqrt(1 / (2 - 1 / (sig1_naive * sig1_naive)));
			double c2 = -(1 / (2 - 1 / (sig1_naive * sig1_naive))) * sqrt(1 / (2 - 1 / (sig1_naive * sig1_naive))) / pow(sig1_naive, 3);
			double proj_taylor1 = (a - c1 * c2 + pow(c2, 2) * sig1_naive + b * c2) / (1 + pow(c2, 2));
			double proj_taylor2 = c1 + (proj_taylor1 - sig1_naive) * c2;

            //setting the matrix according to now decided sigma
			double sig = max(proj_taylor1, proj_taylor2);
			double sig2 = min(proj_taylor1, proj_taylor2);
			if (a < b) {
				sig = min(proj_taylor2, proj_taylor1);
				sig2 = max(proj_taylor1, proj_taylor2);
			}
            Matrix2d sigma;//fill into matrix
			sigma << sig, 0, 0, sig2;

            //fit V (rotation matrix) using Procrustes
            Matrix2d V_fit, S_svd, U_svd, V_svd;
			SSVD2x2(sigma * U.transpose() * Juv, U_svd, S_svd, V_svd);
			V_fit = V_svd * U_svd.transpose();
			Eigen::Matrix2d flip = Eigen::Matrix2d::Identity();
			flip(1, 1) = -1.;
			if (V_fit.determinant() < 0) {//no flipping
				V_fit = V_svd * flip * U_svd.transpose();
			}

            //construct Cheby Jacobian
			Vs[f] = V_fit;
			Ss[f] = sigma;
			Us[f] = U;
            Ri = U * sigma * V_fit.transpose();
            //fill into vector
            R(f) = Ri(0, 0);
            R(f + F.rows()) = Ri(0, 1);
            R(f + 2 * F.rows()) = Ri(1, 0);
            R(f + 3 * F.rows()) = Ri(1, 1);
        }
        first_solve=false;
        mat.clear();
        mat.resize(4);
        mat[0].push_back(Dx);
        mat[0].push_back(dzero);
        mat[1].push_back(Dy);
        mat[1].push_back(dzero);
        mat[2].push_back(dzero);
        mat[2].push_back(Dx);
        mat[3].push_back(dzero);
        mat[3].push_back(Dy);
        igl::cat(mat, A);
        SparseMatrix<double> M, Mass;
        VectorXd area;
        igl::doublearea(V, F, area);
        M = area.asDiagonal();
        vector<vector<SparseMatrix<double>>> matrices2;
        matrices2.resize(4);
        SparseMatrix<double> zero;
        zero.resize(F.rows(), F.rows());
        zero.setZero();
        matrices2[0].push_back(M);
        matrices2[0].push_back(zero);
        matrices2[0].push_back(zero);
        matrices2[0].push_back(zero);
        matrices2[1].push_back(zero);
        matrices2[1].push_back(M);
        matrices2[1].push_back(zero);
        matrices2[1].push_back(zero);
        matrices2[2].push_back(zero);
        matrices2[2].push_back(zero);
        matrices2[2].push_back(M);
        matrices2[2].push_back(zero);
        matrices2[3].push_back(zero);
        matrices2[3].push_back(zero);
        matrices2[3].push_back(zero);
        matrices2[3].push_back(M);
        igl::cat(matrices2, Mass);
        b = A.transpose() * Mass * R;
        A = (A.transpose() * Mass * A).eval();
    }

    // Solve the linear system.
    SparseMatrix<double> sys;
    VectorXd rhs;
    rhs.resize(2 * V.rows() + 2 * con);
    rhs << b, d;
    SparseMatrix<double> z = SparseMatrix<double>(2 * con, 2 * con);
    z.setZero();
    vector<vector<SparseMatrix<double>>> matrices;
    matrices.resize(2);
    matrices[0].push_back(A);
    matrices[0].push_back(C.transpose());
    matrices[1].push_back(C);
    matrices[1].push_back(z);
    igl::cat(matrices, sys);

    SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solver;
    solver.analyzePattern(sys);
    solver.factorize(sys);
    VectorXd sol = solver.solve(rhs);
    UV.resize(V.rows(), 2);
    UV.col(0) = sol.block(0, 0, V.rows(), 1);
    UV.col(1) = sol.block(V.rows(), 0, V.rows(), 1);

    param_3d.resize(UV.rows(), 3);
    param_3d.col(0) = UV.col(0);
    param_3d.col(1) = UV.col(1);
    param_3d.col(2).setConstant(0);
    igl::cotmatrix(param_3d, F, L_orig);
    igl::cotmatrix(param_3d, F, L_curr);
    V_def = param_3d;
    F_tex = F;
    igl::cotmatrix(param_3d, F, L_uv);
    change_uv_not_fac = true;
}

//per-triangle arap deformation
void findRotations_triangles(const Eigen::MatrixXd& V0, const Eigen::MatrixXd& V1, const Eigen::MatrixXi& F, const Eigen::MatrixXi& F_tex, const Eigen::MatrixXd& C, std::vector<Eigen::Matrix3d>& rot) {
    const auto m = F.rows();//triangles

    rot.clear();
    rot.resize(m, Eigen::Matrix3d::Zero());

   for (int i = 0; i < m; i++) {
        Matrix3d temp, temp2;
        temp2.setZero();
        for (int j = 0; j < 3; ++j) {//go over TWO all edges surrounding faces
            const int k1 = j == 2 ? 0 : j + 1; //k1 is wrap-around successor of j
            const int k2 = k1 == 2 ? 0 : k1 + 1;//k2 is wrap-around successor of k1 (2nd to j)
            const Eigen::Vector3d e0 = V0.row(F_tex(i, k1)) - V0.row(F_tex(i, k2));//edge from face i k1 to k2 (goes over all edges due to j)
            const Eigen::Vector3d e1 = V1.row(F(i, k1)) - V1.row(F(i, k2));//same in current setting
            const Eigen::Matrix3d r = C(i, j) * e0 * e1.transpose();//weigh by cotangent weight//NOTE C(i,j)
            rot[i] += r;//add onto rotation matrix of triangle
        }
    }

    // compute optimal rotations
    Eigen::Matrix3d flip = Eigen::Matrix3d::Identity();
    flip(2, 2) = -1.;

    for (int i = 0; i < m; ++i) {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(rot[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
        rot[i] = (svd.matrixV() * svd.matrixU().transpose()).transpose();
        if (rot[i].determinant() < 0) {
            rot[i] = (svd.matrixV() * flip * svd.matrixU().transpose()).transpose();
        }
    }

}

Eigen::MatrixXd rhs_triangles(
    const Eigen::MatrixXd& V,
    const std::vector<std::vector<Edge>>& eSets,
    const std::vector<Eigen::Matrix3d>& R,
    Eigen::MatrixXd& rhs) {
    rhs.setZero();
    int m = F_tex.rows();
    MatrixXd grad(2 * m, 3);
    Matrix3d id;
    id << cos(-PI / 4), -sin(-PI / 4), 0, sin(-PI / 4), cos(-PI / 4), 0, 0, 0, 1;
    for (int i = 0; i < m; i++) {//go over triangles
        grad.row(i) = R[i].row(0);
        grad.row(m + i) = R[i].row(1);
    }
    rhs = J.transpose() * Mass * grad;
    return rhs;
}

//cheby deformation
double compute_cheby_energy(MatrixXd V, MatrixXd V_def, std::vector<MatrixXd> J_ch, std::vector<MatrixXd> old_J_ch) {
    return 0;
    double tr_energy = 0;
    MatrixXd Gx(2, 3);
    MatrixXd Duv = Du * V_def;
    MatrixXd Dvv = Dv * V_def;
    for (int i = 0; i < F.rows(); ++i) {//go over vertices
        Gx.row(0) = Duv.row(i);//should be 2x3
        Gx.row(1) = Dvv.row(i);//should be 2x3
        double tr_term = triangle_area(i) * (J_ch[i] - Gx).squaredNorm();
        double tr_term_old = triangle_area(i) * (old_J_ch[i] - Gx).squaredNorm();
        tr_energy += tr_term;
    }
    return tr_energy;
}

void local_step(const Eigen::MatrixXd& V1, const Eigen::MatrixXi& F, std::vector<Eigen::MatrixXd>& cheby_grad) {//projection to cheby jacobian
    const auto n = V1.rows();//vertices
    const auto m = F.rows();//triangles
    cheby_grad.clear();
    cheby_grad.resize(m);
    MatrixXd Duv = Du * V_def;
    MatrixXd Dvv = Dv * V_def;
    MatrixXd shearing_max_template(3, 2);
    shearing_max_template.col(1) << 1, 0, 0;
    VectorXd degs(1);
    Vector3d dk;
    dk << 0, 1, 0;
    for (int i = 0; i < m; i++) {
        MatrixXd Gx(2, 3);
        Gx.row(0) = Duv.row(i);//should be 1x3
        Gx.row(1) = Dvv.row(i);//should be 1x3
        cheby_grad[i] = Gx.rowwise().normalized();
        if (limit_shearing < 90) {//hard limit on shearing in projection step
            double alpha = acos(cheby_grad[i].row(0).dot(cheby_grad[i].row(1))) / PI * 180;
            double shear = min(alpha, 180 - alpha);
            if (shear < limit_shearing) {//too much shearing, set to the closest possible matrix
                if (alpha < 90) {
                    degs(0) = (limit_shearing / 180 * PI);
                }
                else {
                    degs(0) = ((180 - limit_shearing) / 180 * PI);
                }
                Matrix2d rot_by_deg;
                rot_by_deg << cos(degs(0)), -sin(degs(0)), sin(degs(0)), cos(degs(0));
                Vector2d unit;
                unit << 1, 0;
                Vector2d snd_2 = rot_by_deg * unit;
                Vector3d snd;
                snd << snd_2(0), snd_2(1), 0;
                shearing_max_template.col(0) = snd.normalized();
                Matrix3d rt = Duv.row(i).transpose() * shearing_max_template.col(0).transpose() + Dvv.row(i).transpose() * shearing_max_template.col(1).transpose();
                Vector3d nrm1 = Duv.row(i).transpose();
                Vector3d nrm3 = Dvv.row(i).transpose();
                Vector3d nrm2 = shearing_max_template.col(0);
                Vector3d nrm4 = shearing_max_template.col(1);
                rt += nrm1.cross(nrm2) * nrm3.cross(nrm4).transpose();
                Eigen::Matrix3d flip = Eigen::Matrix3d::Identity();
                flip(2, 2) = -1.;

                for (int i = 0; i < n; ++i) {
                    Eigen::JacobiSVD<Eigen::Matrix3d> svd(rt, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    rt = svd.matrixV() * svd.matrixU().transpose();
                    if (rt.determinant() < 0) {
                        rt = svd.matrixV() * flip * svd.matrixU().transpose();
                    }
                }
                cheby_grad[i] = (rt * shearing_max_template).transpose();
                MatrixXd rotated = (rt * shearing_max_template).transpose();
            }
        }
    }
}

Eigen::MatrixXd rhs_cheby(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<Eigen::MatrixXd>& J_ch,
    Eigen::MatrixXd& rhs) {

    const int n = V.rows();
    const int m = F.rows();
    MatrixXd proj(2 * m, 3);

    for (int i = 0; i < m; i++) {//go over triangles
        proj.row(i) = J_ch[i].row(0);
        proj.row(m + i) = J_ch[i].row(1);
    }
    rhs = J.transpose() * Mass * proj;
    return rhs;
}


//system solving steps, core functions calling the respective methods
void factorize(Viewer& viewer, double lambda) {
    if (num_handles == 0) {
        return;
    }
    was_in.resize(V.rows());

    //redo for param basis
    igl::adjacency_list(F, adj_list);
    igl::cotmatrix_entries(param_3d, F_tex, C_tr);//for rotation fitting arap
    igl::cotmatrix(V_orig, F, L);//write into global L
    Eigen::SparseMatrix<double> sys, solver_mat;
    igl::massmatrix(V_orig, F, igl::MASSMATRIX_TYPE_VORONOI, M);
    igl::doublearea(UV, F_tex, triangle_area);
    triangle_area /= 2;
    M_tr = triangle_area.asDiagonal();
    SparseMatrix<double> M_zero(F.rows(), F.rows());
    M_zero.setZero();
    vector<vector<SparseMatrix<double>>> vecmat;
    vecmat.clear();
    vecmat.resize(2);
    vecmat[0].push_back(M_tr);
    vecmat[0].push_back(M_zero);
    vecmat[1].push_back(M_zero);
    vecmat[1].push_back(M_tr);
    igl::cat(vecmat, Mass);

    //construct system matrix 
    //TODO add mass
    computeSurfaceGradientMatrix(UV, F_tex, V_orig, F, Du, Dv);
    igl::cat(1, Du, Dv, J);//note J global to access in rhs
    //Laplacians for bending regularization
    //uv x uv we have L_uv, but since we're taking the derivative wrt V, we need to adapt (sum over corresponding uv)
    //this translating matrix is L_pl. shouldn't matter since multiplication of matrices is additive (can do before or after)
    L_pl.resize(UV.rows(), V_orig.rows());
    L_pl.setZero();
    L_pl.reserve(L_uv.nonZeros());
    //fill L_pl
    vector<vector<bool>> donee;//UV x V like Lpl itself. check if coeff already filled or no
    donee.resize(UV.rows());
    for (int i = 0; i < donee.size(); i++) {
       donee[i] = vector<bool>(V_orig.rows(), false);
    }
    MatrixXd rot_laps_bending = L_uv * param_3d;
    for (int u = 0; u < UV.rows(); u++) {
        if(rot_laps_bending.row(u).norm()<0.00001){//ignore all borders (also includes seams)
            for (SparseMatrix<double>::InnerIterator it(L_uv, u); it; ++it)
            {
                if (donee[u][uv_to_v_corr[it.row()]]) {
                    L_pl.coeffRef(u, uv_to_v_corr[it.row()]) += it.value();
                }
                else {
                    donee[u][uv_to_v_corr[it.row()]] = true;
                    L_pl.insert(u, uv_to_v_corr[it.row()]) = it.value();
                }
            }
        }
    }
    
    v_to_uv_corr.resize(V_orig.rows());
    //build reverse index
    for (int p = 0; p < param_3d.rows(); p++) {
       v_to_uv_corr[uv_to_v_corr[p]].insert(p);
    }

    sys = (1 - no_bend) * (J.transpose() * Mass * J) +no_bend * (L_pl.transpose() * L_pl) / (L_pl.transpose() * L_pl).norm() * (J.transpose() * Mass * J).norm();
    if (use_gravity) {
        dyn_dw = 1 / dyn_ym * dyn_h * dyn_h;
        SparseMatrix<double> DQ = dyn_dw * 1 / (dyn_h * dyn_h) * M;
        dq_norm = DQ.norm();
        sys += DQ;
        dyn_vel = V_orig;
        dyn_vel.setZero();
        dyn_f_ext = V_orig;
        dyn_f_ext.setZero();
        dyn_f_ext.col(1).setConstant(dyn_force);
        collision_out_weights.resize(V.rows());
        collision_out_weights.setConstant(1 / dyn_h * dyn_ym / dq_norm);
    }

    //constrain system
    int num_free = V_orig.rows() - handle_vertices.size();
    v_free_index.resize(num_free);
    v_constrained_index = handle_vertices;
    int count_free = 0;
    for (int i = 0; i < handle_id.size(); ++i) {
        if (handle_id[i] == -1) {
            v_free_index[count_free++] = i;
        }
    }
    igl::slice(sys, v_free_index, v_free_index, solver_mat);
    igl::slice(sys, v_free_index, v_constrained_index, free_influenced);
    solver.compute(solver_mat);
    change_uv_not_fac = false;
}

bool solve(Viewer& viewer) {
    if (num_handles == 0) {
        return false;
    }
    if (change_uv_not_fac) {
        factorize(viewer, lambda);
    }
    igl::slice_into(handle_vertex_positions, handle_vertices, 1, V_def);
    Eigen::MatrixXd b2 = free_influenced * handle_vertex_positions;//constraints
    Eigen::MatrixXd b;//vector to solve for

    //chebyshev deformation method
    std::vector<Eigen::MatrixXd> J_ch;
    local_step(V_def, F, J_ch);
    old_cheby_grad = J_ch;
    Eigen::MatrixXd b_ch;
    rhs_cheby(V, F, J_ch, b_ch);
    b = (1 - lambda) * b_ch;
    if (lambda) {//arap regularization
        std::vector<Eigen::Matrix3d> R;
        findRotations_triangles(param_3d, V, F, F_tex, C_tr, R);    
        MatrixXd b_arap;
        rhs_triangles(param_3d, edgeSets_tr, R, b_arap);
        b += lambda * b_arap;
    }
    if (no_bend) {//bending regularization
        MatrixXd rot_laps_bending = L_uv * param_3d;
        b = (1 - no_bend) * b + no_bend / (L_pl.transpose() * L_pl).norm() * (J.transpose() * Mass * J).norm() * L_pl.transpose() * rot_laps_bending;
    }
    MatrixXd U0;
    if (use_gravity) {//gravity and dynamics term
        U0 = V_def;
        MatrixXd D1 = dyn_dw * (1 / (dyn_h * dyn_h) * M * (-U0 - dyn_h * dyn_vel) - dyn_f_ext);
        b -= D1;
    }
    //solve for b in a constrained (handle positions) manner
    Eigen::MatrixXd bI;
    igl::slice(b, v_free_index, 1, bI);//only non-handle part
    Eigen::MatrixXd V_free_deformed = solver.solve(bI - b2);//solve constrained
    //put result into V_def
    V_def = V_orig;
    igl::slice_into(V_free_deformed, v_free_index, 1, V_def);
    igl::slice_into(handle_vertex_positions, handle_vertices, 1, V_def);
    //compute energy
    double curr_energy = compute_cheby_energy(UV, V_def, J_ch, J_ch);//don't care about old grad here, just going in local
    actual_energy = curr_energy;
    if (use_gravity) {//update parameters for dynamics
        dyn_vel = (V_def - U0) / dyn_h;
    }

    //V holds old, V_def new
    if (use_collision && use_gravity) {//do collision handling (only active with gravity)
        MatrixXd S_c, I_c, C_c, N_c;
        igl::signed_distance(V_def, V_collision_plus, F_collision, igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL, S_c, I_c, C_c, N_c);
        for (int i = 0; i < V.rows(); i++) {
            igl::Hit H;
            if (S_c(i) > 0) {//not inside
                if (was_in[i]) {//was in before, so adapt weight of pushing out term 
                    dyn_vel.row(i) /= 1.25;
                    collision_out_weights(i) /= 1.25;

                }
                else {//was not in before, set to default
                    dyn_vel.row(i).setZero();
                    collision_out_weights(i) = 1 / dyn_h * dyn_ym / dq_norm;
                }
                was_in[i] = false;
            }
            else {//collison detected
                dyn_vel.row(i) = (C_c.row(i) - V_def.row(i)) * collision_out_weights(i);//bounce off
                if (collision_out_weights(i) < 10000) {//below cutoff value
                    collision_out_weights(i) *= 1.5;//higher weight to make sure it will not stay inside mesh
                }
                was_in[i] = true;
            }
        }
    }

    //update mesh
    V = V_def;
    viewer.data(0).set_vertices(V);
    viewer.data(0).compute_normals();
    viewer.data(0).show_lines = false;
    viewer.data().clear_points();
    viewer.data().add_points(handle_vertex_positions, point_color);
    viewer.data(0).set_face_based(false);


    return true;
};

int main(int argc, char* argv[]) {
    // Load a mesh from file
    std::string filename, coll_filename;
    if (argc >= 2) {
        filename = "../data/" + std::string(argv[1]); // Mesh provided as command line argument
    }
    else {
        filename = "../data/scaled_cyl.obj"; // Default mesh
    }
    //read in all meshes and initialize different global parameters
    viewer.load_mesh_from_file(filename);
    igl::read_triangle_mesh(filename, V, F);
    //collision mesh 
    if (argc >= 3) {
        coll_filename = "../data/" + std::string(argv[2]); // Mesh provided as command line argument
        igl::read_triangle_mesh(coll_filename, V_collision, F_collision);
        V_collision_plus=V_collision;
        viewer.load_mesh_from_file(coll_filename);
    }
    else{
        igl::read_triangle_mesh("../data/draping_mannequin.obj", V_collision, F_collision);
        V_collision_plus=V_collision;
        viewer.load_mesh_from_file("../data/draping_mannequin.obj");
    }
    
    F_tex = F;
    V_orig = V;
    V_def = V;
    viewer.data(0).set_mesh(V, F);
    if (use_collision) {
        viewer.data(1).set_mesh(V_collision, F_collision);
    }
    else{
        viewer.data(1).clear();
    }
    temp3D = viewer.core();
    temp2D = viewer.core();
    temp2D.orthographic = true;
    per_face_energy.resize(F.rows());
    per_face_energy.fill(INFINITY);

    //precomputations on mesh
    igl::adjacency_list(F, adj_list);
    igl::cotmatrix(V_orig, F, L);//write into global L
    igl::per_face_normals(V_collision, F_collision, FN_collision);
    igl::AABB<Eigen::MatrixXd, 3> tree;
    tree.init(V, F);
    and_visible =
        Eigen::Array<double, Eigen::Dynamic, 1>::Zero(V.rows());
    old_cheby_grad.clear();
    old_cheby_grad.resize(F.rows(), Eigen::MatrixXd(2, 3));
    for (int i = 0; i < F.rows(); i++) {
        old_cheby_grad[i].setZero();
    }

    //initialize handles 
    //id to -1 because nothing was assigned yet
    handle_id.setConstant(V.rows(), 1, -1);
    //handle plugins for deformation, selection, menu
    igl::opengl::glfw::imgui::ImGuiPlugin imgui_plugin;
    viewer.plugins.push_back(&imgui_plugin);
    // Add a 3D gizmo plugin
    guizmo.operation = ImGuizmo::TRANSLATE;
    imgui_plugin.widgets.push_back(&guizmo);
    guizmo.visible = false;
    guizmo.T.block(0, 3, 3, 1) = V.row(plugin_vertex).transpose().cast<float>();

    //add menu
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    imgui_plugin.widgets.push_back(&menu);
    //Add selection plugin
    selection.mode = igl::opengl::glfw::imgui::SelectionWidget::OFF;
    imgui_plugin.widgets.push_back(&selection);

    // Attach callback to apply imguizmo's transform to mesh
    guizmo.callback = [&](const Eigen::Matrix4f& T)
    {
        MatrixXd S_c, I_c, C_c, N_c;
        MatrixXd old_pluginpos(3,1);
        old_pluginpos=pluginpos;
        MatrixXf orig_T0 = T0;
        T0.block(0, 3, 3, 1) = pluginpos.transpose().cast<float>();
        bool in = false;
        const Eigen::Matrix4d TT = (T * T0.inverse()).cast<double>().transpose();

        pluginpos = ((pluginpos.rowwise().homogeneous() * TT).rowwise().hnormalized()).eval();//update pluginpos
        MatrixXd old_handle_pos = handle_vertex_positions;
        MatrixXd old_V = V;
        V_def = (V.rowwise().homogeneous() * TT).rowwise().hnormalized();
        int id = handle_id(plugin_vertex);
        if (handle_vertex_positions.rows() > 0) {
            for (int i = 0; i < handle_vertices.rows(); i++) {
                if (handle_id(handle_vertices(i)) == id) {//this handle
                    handle_vertex_positions.row(i) = V_def.row(handle_vertices(i));//update handle pos
                }
            }
            if (use_collision) {
                igl::signed_distance(handle_vertex_positions, V_collision, F_collision, igl::SIGNED_DISTANCE_TYPE_PSEUDONORMAL, S_c, I_c, C_c, N_c);
                for (int i = 0; i < handle_vertices.rows(); i++) {
                    if (S_c(i) < 0) {//some constraint inside
                        in = true;
                        handle_vertex_positions.row(i) = C_c.row(i);
                    }
                }
            }
        }

        T0 = T;//update transform
        solve(viewer);//whenever the guizmo is active, we solve (interactively)
    };

    //selection plugin
    selection.callback = [&]()
    {
        igl::screen_space_selection(V, F, tree, viewer.core().view, viewer.core().proj, viewer.core().viewport, selection.L, sel_vertices, and_visible);
        make_area_handle();
    };

    //draw menu
    menu.callback_draw_viewer_menu = [&]()
    {
        if (ImGui::CollapsingHeader("Deformation Controls", ImGuiTreeNodeFlags_DefaultOpen))
        {
            int trans_type = static_cast<int>(transform_mode);
            if (ImGui::Combo("Transformation Mode", &trans_type, "ROTATE\0TRANSLATE\0SCALE\0"))
            {
                transform_mode = static_cast<Trans>(trans_type);
                if (transform_mode == TRANSLATE) {
                    viewer.callback_key_pressed(viewer, 't', 0);
                }
                if (transform_mode == ROTATE) {
                    viewer.callback_key_pressed(viewer, 'r', 0);
                }
                if (transform_mode == SCALE) {
                    guizmo.operation = ImGuizmo::SCALE;
                }
            }
            int handle_type = static_cast<int>(handle_option);
            if (ImGui::Combo("Handle Option", &handle_type, "LASSO\0MARQUE\0VERTEX\0REMOVE\0NONE\0"))
            {
                handle_option = static_cast<Handle>(handle_type);
                if (handle_option == LASSO) {
                    selection.mode = igl::opengl::glfw::imgui::SelectionWidget::LASSO;
                    viewer.callback_key_pressed(viewer, 'l', 0);
                }
                if (handle_option == MARQUE) {
                    selection.mode = igl::opengl::glfw::imgui::SelectionWidget::RECTANGULAR_MARQUEE;
                    viewer.callback_key_pressed(viewer, 'm', 0);
                }
                if (handle_option == VERTEX) {
                    viewer.callback_key_pressed(viewer, 'p', 0);
                }
                if (handle_option == NONE) {
                    viewer.callback_key_pressed(viewer, 'v', 0);
                }
                if (handle_option == REMOVE) {
                    viewer.callback_key_pressed(viewer, 'x', 0);
                }
            }
            if (ImGui::InputDouble("ARAP regularization [0,1)", &lambda, 0, 0)) {
                factorize(viewer, lambda);
                solve(viewer);
            }
            if (ImGui::InputDouble("limit shearing [0,90]", &limit_shearing, 0, 0)) {
                solve(viewer);
            }
            if (ImGui::InputDouble("Bending regularization [0,1)", &no_bend, 0, 0)) {
                factorize(viewer, lambda);
                solve(viewer);
            }
            if (ImGui::InputDouble("Texture Resolution", &TextureResolution, 0, 0)) {
                Redraw();
            }
            if (ImGui::Checkbox("Show Texture", &show_texture)) {
                Redraw();
            }
            if (ImGui::Checkbox("Collision", &use_collision)) {
                if (use_collision) {
                    viewer.data(1).set_mesh(V_collision, F_collision);
                    Eigen::MatrixXd col2(V_collision.rows(), 4);
                    for (int i = 0; i < V_collision.rows(); i++)//go over vertices
                    {
                        col2.row(i) << 1, 1, 1, 0;
                    }
                    viewer.data(1).set_colors(col2);
                }
                else {
                    viewer.data(1).clear();
                }
            }
            
            if (ImGui::Checkbox("Gravity", &use_gravity)) {
                factorize(viewer, lambda);
                solve(viewer);
            }
            if (ImGui::InputDouble("Young's modulus", &dyn_ym, 0, 0)) {
                factorize(viewer, lambda);
                solve(viewer);
            }
            if (ImGui::InputDouble("Timestep", &dyn_h, 0, 0)) {
                factorize(viewer, lambda);
                solve(viewer);
            }
            if (ImGui::InputDouble("External force", &dyn_force, 0, 0)) {
                dyn_f_ext = V_orig;
                dyn_f_ext.setZero();
                dyn_f_ext.col(1).setConstant(dyn_force);
                solve(viewer);
            }
            if (ImGui::Button("Save .obj", ImVec2(-1, 0)))
            {
                std::fstream s{ "../res/textured.obj", s.binary | s.trunc | s.in | s.out };
                for (int i = 0; i < V_orig.rows(); i++) {
                    s << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << std::endl;
                }
                for (int i = 0; i < UV.rows(); i++) {
                    s << "vt " << UV(i, 0) << " " << UV(i, 1) << std::endl;
                }
                for (int i = 0; i < F.rows(); i++) {
                    s << "f " << F(i, 0) + 1 << "/" << F_tex(i, 0) + 1 << " "
                        << F(i, 1) + 1 << "/" << F_tex(i, 1) + 1 << " "
                        << F(i, 2) + 1 << "/" << F_tex(i, 2) + 1 << " " << std::endl;
                }
                s.close();
            }
            if (ImGui::Button("Load texture from .obj", ImVec2(-1, 0)))
            {
                //for pre-parametrized meshes
                MatrixXd VT, CN, FN;
                MatrixXi FT;
                igl::readOBJ(filename, V, VT, CN, F, FT, FN);
                UV = VT;
                F_tex = FT;
                param_3d.resize(UV.rows(), 3);
                param_3d.col(0) = UV.col(0);
                param_3d.col(1) = UV.col(1);
                param_3d.col(2).setConstant(0);
                igl::cotmatrix(param_3d, F_tex, L_orig);
                igl::cotmatrix(param_3d, F_tex, L_curr);
                viewer.data(0).set_uv(TextureResolution*UV, F_tex);
                viewer.data(0).show_texture = show_texture;
                viewer.data(0).set_face_based(false);
                igl::cotmatrix(UV, F_tex, L_uv);
                // Read the PNG
                //Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R,G,B,A;
                //igl::stb::read_image("../data/texture-silk.png",R,G,B,A);
                //viewer.data(0).set_uv(TextureResolution * UV, F_tex);
                //viewer.data(0).show_texture = show_texture;
                //viewer.data(0).set_texture(R,G,B);

                change_uv_not_fac = true;
            }
            if (ImGui::Button("Recenter on fabric", ImVec2(-1, 0))) {
                viewer.core().align_camera_center(V);
            }
        }
    };

    // Maya-style keyboard shortcuts for operation
    viewer.callback_key_pressed = [&](decltype(viewer)&, unsigned int key, int mod)
    {
        vertex_picking_mode = false;
        handle_deleting_mode = false;
        switch (key) {
        case 'T': case 't': guizmo.operation = ImGuizmo::TRANSLATE;  transform_mode = TRANSLATE;  return true;
        case 'R': case 'r': guizmo.operation = ImGuizmo::ROTATE;  transform_mode = ROTATE;  return true;
        case 'V': case 'v': vertex_picking_mode = false; handle_option = NONE;  return true;
        case 'l': handle_option = LASSO;  return true;
        case 'P': case 'p': vertex_picking_mode = true; handle_option = VERTEX;  selection.mode = igl::opengl::glfw::imgui::SelectionWidget::OFF; return true;//try to add vertex picking mode 
        case 'X': case 'x': handle_deleting_mode = true; handle_option = REMOVE; return true;
        case '1': computeParameterization(1); Redraw(); return true;
        case '2': computeParameterization(2); Redraw(); return true;
        case ' ': // space bar -  switches view between mesh and parameterization
            if (!showingUV)
            {
                if (UV.rows() > 0){
                    viewer.data(0).clear();
                    viewer.data(0).set_mesh(UV, F_tex);
                    viewer.core().align_camera_center(UV);
                    viewer.data(0).set_colors(mesh_color);
                    viewer.data(0).set_uv(TextureResolution * UV, F_tex);
                    showingUV = true;
                    viewer.data(0).set_face_based(false);
                    viewer.data(0).show_texture = show_texture;
                }
                else { std::cout << "ERROR ! No valid parameterization\n"; }
            }
            else
            {
                viewer.data(0).clear();
                viewer.data(0).set_mesh(V, F);
                viewer.core().align_camera_center(V);
                viewer.data(0).show_lines = false;
                viewer.data(0).set_colors(mesh_color);
                viewer.data(0).set_face_based(false);
                showingUV = false;
                if (UV.rows() > 0){
                    // Read the PNG
                    Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R,G,B,A;
                    igl::stb::read_image("../data/silk_texture.png",R,G,B,A);
                    viewer.data(0).set_uv(TextureResolution * UV, F_tex);
                    viewer.data(0).show_texture = show_texture;
                    viewer.data().set_texture(R,G,B);
                    /* if (UV.size() != 0) {//nicer design for garments than checkerboard pattern:)
                    viewer.data(0).set_uv(TextureResolution * UV, F_tex);
                    Matrix<unsigned char, 30, 30> R; 
                    Matrix<unsigned char, 30, 30> G;
                    Matrix<unsigned char, 30, 30> B;
                    Matrix<unsigned char, 30, 30> A;
                    R.setConstant(255 * mesh_color[0]);
                    R.col(0).setConstant(255);
                    R.col(15).setConstant(255);
                    R.row(0).setConstant(255);
                    R.row(15).setConstant(255);
                    G.setConstant(255 * mesh_color[1]);
                    G.col(0).setConstant(255);
                    G.col(15).setConstant(255);
                    G.row(0).setConstant(255);
                    G.row(15).setConstant(255);
                    B.setConstant(255 * mesh_color[2]);
                    B.col(0).setConstant(255);
                    B.col(15).setConstant(255);
                    B.row(0).setConstant(255);
                    B.row(15).setConstant(255);
                    A.setConstant(255);
                    viewer.data(0).show_texture = true;//show_texture;
                    viewer.data(0).set_texture(R, G, B, A);
                } */
                }
            }

            return true;
        case 'C': case 'c':
        {
            should_decrease = true;
            V_def = V;
            for (int i = 0; i < 5; i++) {
                solve(viewer);
            }
            should_decrease = false;
            return true;
        }
        }
        return false;
    };

    //if vertex picking mode will add handle
    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        if (true) {
            int fid;
            Eigen::Vector3f bc;
            // Cast a ray in the view direction starting from the mouse position
            double x = viewer.current_mouse_x;
            double y = viewer.core().viewport(3) - viewer.current_mouse_y;
            if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core().view,
                viewer.core().proj, viewer.core().viewport, V, F, fid, bc))
            {
                if (!(vertex_picking_mode || handle_deleting_mode)) {
                    return false;
                }
                float max = bc.maxCoeff();//point that is closest
                int point_face_idx = 0;//find argmax
                for (int i = 0; i < 3; i++) {
                    if (bc(i) == max) {
                        point_face_idx = i;
                    }
                }
                int point = F(fid, point_face_idx);//indexes into V
                if (!handle_deleting_mode) { //now add that vertex to handle
                    //check if already exists
                    if (handle_id(point) != -1) {
                        plugin_vertex = point;
                        guizmo.visible = true;
                        compute_handle_centroid(pluginpos, handle_id(point));
                        guizmo.T.block(0, 3, 3, 1) = pluginpos.transpose().cast<float>();//V.row(plugin_vertex).transpose().cast<float>();//position
                        return true;
                    }
                    //is new vertex
                    Eigen::VectorXd pt = V.row(point);
                    viewer.data().add_points(V.row(point), point_color);
                    num_handles++;
                    //plugin_vertex_index = num_handles - 1;
                    Eigen::VectorXi up_handle_vertices = Eigen::VectorXi(num_handles);//update handle vertex vector
                    Eigen::MatrixXd up_handle_pos = Eigen::MatrixXd(num_handles, 3);//update handle vertex vector
                    for (int i = 0; i < num_handles - 1; i++) {
                        up_handle_vertices(i) = handle_vertices(i);
                        up_handle_pos.row(i) = handle_vertex_positions.row(i);
                    }
                    up_handle_vertices(num_handles - 1) = point;
                    up_handle_pos.row(num_handles - 1) = V.row(point);
                    handle_vertices = up_handle_vertices;
                    handle_vertex_positions = up_handle_pos;
                    plugin_vertex = point;
                    handle_id(plugin_vertex) = plugin_vertex;//set id to itself
                    factorize(viewer, lambda);
                    //now make plugin 'active' at that location
                    guizmo.visible = true;
                    pluginpos = V.row(point);
                    guizmo.T.block(0, 3, 3, 1) = pluginpos.transpose().cast<float>();//position
                }
                else {
                    vector<int> handle_to_delete_idx;
                    bool found = false;
                    for (int i = 0; i < num_handles; i++) {
                        if (handle_id(handle_vertices(i)) == handle_id(point)) {//check if selected handle is at index i in the handle vector
                            handle_to_delete_idx.push_back(i);
                            found = true;
                        }
                    }
                    if (!found) {
                        cout << "no handle to delete" << endl;
                        return false;
                    }
                    else {
                        for (int i = 0; i < handle_to_delete_idx.size(); i++) {
                            handle_id(handle_vertices(handle_to_delete_idx[i])) = -1;//does not belong to handle anymore
                        }
                        num_handles -= handle_to_delete_idx.size();
                        Eigen::VectorXi up_handle_vertices = Eigen::VectorXi(num_handles);//update handle vertex vector
                        Eigen::MatrixXd up_handle_pos = Eigen::MatrixXd(num_handles, 3);//update handle vertex vector
                        int curr = 0;
                        for (int i = 0; i < num_handles + handle_to_delete_idx.size(); i++) {
                            if (handle_id(handle_vertices(i)) != handle_id(point)) {
                                up_handle_vertices(curr) = handle_vertices(i);
                                up_handle_pos.row(curr) = handle_vertex_positions.row(i);
                                curr++;
                            }
                        }
                        handle_vertices = up_handle_vertices;
                        handle_vertex_positions = up_handle_pos;
                        if (num_handles > 0) {
                            factorize(viewer, lambda);
                            //solve(viewer);
                        }
                        else {
                            V = V_orig;
                            viewer.data(0).set_vertices(V);
                        }
                    }
                    viewer.data().clear_points();
                    viewer.data().add_points(handle_vertex_positions, point_color);
                }

                return true;
            }
            return false;
        }
    };

    //display shortcut keys
    std::cout << R"(
T,t   Switch to translate operation
R,r   Switch to rotate operation
P, p  Click to select a vertex
X, x  Click on handle to remove
M, m  Area marquee selection
l     Area lasso selection
1     Compute lscm parametrization (initialize)
2     Compute cheby parametrization
)";

    //set up viewer
    Eigen::MatrixXd col2(V_collision.rows(), 4);
    for (int i = 0; i < V_collision.rows(); i++)//go over vertices
    {
        col2.row(i) << 1, 1, 1, 0;
    }
    if (use_collision) {
        viewer.data(1).set_colors(col2);
    }
    viewer.data().compute_normals();
    viewer.data(0).show_lines = false;
    viewer.data().point_size = 7;
    viewer.data(0).set_colors(mesh_color);
    viewer.data(0).set_face_based(false);
    viewer.core().background_color.setOnes();
    viewer.launch();
}
