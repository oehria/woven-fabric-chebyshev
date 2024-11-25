#include <igl/read_triangle_mesh.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <igl/local_basis.h>
#include <igl/grad.h>
#include <igl/min_quad_with_fixed.h>
#include <igl/cotmatrix.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/lscm.h>
#include <igl/adjacency_matrix.h>
#include <igl/sum.h>
#include <igl/speye.h>
#include <igl/repdiag.h>
#include <igl/cat.h>
#include <igl/exact_geodesic.h>
#include <igl/vertex_components.h>
#include <igl/unproject_onto_mesh.h>
#include <iostream>
#include <fstream>


using namespace std;
using namespace Eigen;
using Viewer = igl::opengl::glfw::Viewer;

Viewer viewer;

//variables
Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd UV;
MatrixXi Ftex;
SparseMatrix<double> Dx, Dy;
vector<Matrix2d> Vs;
vector<Matrix2d> Ss;
vector<Matrix2d> Us;
Eigen::Matrix2d flip = Eigen::Matrix2d::Identity();
bool first_solve = true;//need to factorize Cheby system
bool first_arap_solve = true;//need to factorize ARAP system
SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> solver;
SparseMatrix<double> M, Mass;
VectorXd area;
SparseMatrix<double> rhs_mult, sys_mat;
VectorXd d;
Eigen::SparseMatrix<double> C;
double thread_angle_limit = 0;
//for fixing points 
bool computed_free = false;
double max_dist = 0;
int v1, v2;
//for constraining points
int comp_idx;
bool fix_warp = false;
bool fix_weft = false;
bool fix_anchor = false;
char* warp_mode = "Constrain warp direction";
char* weft_mode = "Constrain weft direction";
MatrixXd anchor_col, warp_col, weft_col;
vector<int> warp_idxs;
vector<int> weft_idxs;
int anchor_idx = 0;
//metrics where cheby error just defined as squared norm between jac.inv and closest cheby jac (cheby error from paper)
VectorXd per_face_energy;
double global_energy;
vector<double> ad_energy_coll;
vector<double> lg_energy_coll;
double runtime = -7;
double runtime_first = -3;
int max_iters = 500;
bool converged = false;
double convergence_precision = 0.0001;
int num_iters_until_convergence = -1;//in case not computed here
int averaging_runtime_iters = 1;//for times reported in paper we ran multiple experiments to have a more consistent runtime
MatrixXd init;


//viewer
bool showingUV = false;
double TextureResolution = 10;
igl::opengl::ViewerCore temp3D;
igl::opengl::ViewerCore temp2D;
Eigen::RowVector3d mesh_color(146 / 255.0, 187 / 255.0, 227 / 255.0);
string save_name;

//helper functions
void Redraw()
{
	viewer.data().clear();

	if (!showingUV)
	{
		viewer.data().set_mesh(V, F);
		viewer.data(0).set_colors(mesh_color);
		viewer.data().set_face_based(false);
		if (UV.size() != 0)
		{
			viewer.data().set_uv(TextureResolution * UV, Ftex);
			viewer.data().show_texture = true;
		}
	}
	else
	{
		viewer.data().clear_points();
		viewer.data().show_texture = false;
		viewer.data().set_mesh(UV, Ftex);
		viewer.data(0).set_colors(mesh_color);
		viewer.core().align_camera_center(UV);
		viewer.data().add_points(UV.row(anchor_idx), anchor_col * 0.8);
		for (int i = 0; i < warp_idxs.size(); i++) {
			viewer.data().add_points(UV.row(warp_idxs[i]), warp_col * 0.8);
		}
		for (int i = 0; i < weft_idxs.size(); i++) {
			viewer.data().add_points(UV.row(weft_idxs[i]), weft_col * 0.8);
		}
	}
}

bool callback_mouse_move(Viewer& viewer, int mouse_x, int mouse_y)
{
	if (showingUV)
		viewer.mouse_mode = igl::opengl::glfw::Viewer::MouseMode::Translation;
	return false;
}

static void computeSurfaceGradientMatrix(SparseMatrix<double>& D1, SparseMatrix<double>& D2)
{
	MatrixXd F1, F2, F3;
	SparseMatrix<double> DD, Dx, Dy, Dz;

	igl::local_basis(V, F, F1, F2, F3);
	igl::grad(V, F, DD);

	Dx = DD.topLeftCorner(F.rows(), V.rows());
	Dy = DD.block(F.rows(), 0, F.rows(), V.rows());
	Dz = DD.bottomRightCorner(F.rows(), V.rows());

	D1 = F1.col(0).asDiagonal() * Dx + F1.col(1).asDiagonal() * Dy + F1.col(2).asDiagonal() * Dz;
	D2 = F2.col(0).asDiagonal() * Dx + F2.col(1).asDiagonal() * Dy + F2.col(2).asDiagonal() * Dz;

	// Define triplet lists for Dx and Dy
	std::vector<Eigen::Triplet<double>> triplets_Dx;
	std::vector<Eigen::Triplet<double>> triplets_Dy;

	// Fill triplets
	for (int f = 0; f < Ftex.rows(); f++) {
		for (int i = 0; i < 3; ++i) {
			int tex_index = Ftex(f, i);
			int vertex_index = F(f, i);
			triplets_Dx.push_back(Eigen::Triplet<double>(f, tex_index, D1.coeff(f, vertex_index)));
			triplets_Dy.push_back(Eigen::Triplet<double>(f, tex_index, D2.coeff(f, vertex_index)));
		}
	}

	// Construct Dx and Dy matrices from triplets
	Dx.resize(Ftex.rows(), UV.rows());
	Dy.resize(Ftex.rows(), UV.rows());
	Dx.setFromTriplets(triplets_Dx.begin(), triplets_Dx.end());
	Dy.setFromTriplets(triplets_Dy.begin(), triplets_Dy.end());

	D1 = Dx;
	D2 = Dy;
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

void ConvertConstraintsToMatrixForm(VectorXi indices, MatrixXd positions, int type, Eigen::SparseMatrix<double>& C, VectorXd& d)
{
	int size = 2 * indices.size() + warp_idxs.size() + weft_idxs.size();//number of constrained entries, anchor in both directions, warp and weft only in one
	if (type == '3' && warp_idxs.size() != 0) {//one warp index was used as fully fixed
		size--;
	}
	C.resize(size, 2 * UV.rows());
	C.reserve(size);
	d.resize(size);
	d.setZero();
	//fix anchor + other point in type 3 case
	int offset = 0;
	for (int i = 0; i < indices.size(); i++) {
		C.insert(2 * i, indices(i)) = 1;//u
		d(2 * i) = positions(i, 0);
		C.insert(2 * i + 1, indices(i) + UV.rows()) = 1;//v
		d(2 * i + 1) = positions(i, 1);
		offset += 2;
	}
	//constrain warp direction only
	int idx = 0;
	for (int i = 0; i < warp_idxs.size(); i++) {
		if (type != '3' || indices(1) != warp_idxs[i]) {
			C.insert(offset + idx, warp_idxs[i] + UV.rows()) = 1;//v fixed bc horizontal
			d(offset + idx) = positions(0, 1);//continue anchor in that direction
			idx++;
		}
	}
	offset += warp_idxs.size();
	if (type == '3' && warp_idxs.size() != 0) {//one warp index was used as fully fixed
		offset--;
	}
	//constrain weft
	for (int i = 0; i < weft_idxs.size(); i++) {
		C.insert(offset + i, weft_idxs[i]) = 1;//u fixed bc vertical
		d(offset + i) = positions(0, 0);//continue anchor in that direction
	}
}

bool load_mesh(string filename)
{
	igl::read_triangle_mesh(filename, V, F);
	Redraw();
	viewer.core().align_camera_center(V);
	showingUV = false;
	per_face_energy.resize(F.rows());
	per_face_energy.fill(INFINITY);
	global_energy = INFINITY;
	igl::doublearea(V, F, area);
	area *= 0.5;
	M=area.asDiagonal();
	return true;
}

bool callback_init(Viewer& viewer)
{
	temp3D = viewer.core();
	temp2D = viewer.core();
	temp2D.orthographic = true;

	return false;
}

//visualization
void cheby_distortion(Eigen::VectorXd& face_dist) {
	//compute the cheby error as defined in the paper
	SparseMatrix<double> Dx, Dy;
	computeSurfaceGradientMatrix(Dx, Dy);
	vector<vector<SparseMatrix<double>>> mat_vec;
	mat_vec.resize(2);
	mat_vec[0].push_back(Dx);
	mat_vec[1].push_back(Dy);
	VectorXd Ju, Jv;
	SparseMatrix<double> J;
	igl::cat(mat_vec, J);
	Ju = J * UV.col(0);
	Jv = J * UV.col(1);
	double max_dist = -100000;
	double min_dist = 1000000;
	face_dist.resize(F.rows());
	for (int f = 0; f < F.rows(); f++) {
		Eigen::Matrix2d Juv;
		Juv << Ju(f), Ju(f + F.rows()), Jv(f), Jv(f + F.rows());
		face_dist(f) = std::log((Juv.inverse() - Juv.inverse().colwise().normalized()).squaredNorm());//show in logarithmic scale so more problematic parts can be seen
		face_dist(f) = (Juv.inverse() - Juv.inverse().colwise().normalized()).norm();
		if (face_dist(f) > max_dist) {
			max_dist = face_dist(f);
		}
		if (face_dist(f) < min_dist) {
			min_dist = face_dist(f);
		}
	}
	//normalize
	face_dist.array() -= min_dist;
	face_dist /= (max_dist - min_dist);

	//show shearing instead	
	/* max_dist = -1000;
	min_dist = 10000;
	for (int f = 0; f < F.rows(); f++) {
		Eigen::Matrix2d Juv;
		Juv << Ju(f), Ju(f + F.rows()), Jv(f), Jv(f + F.rows());
		double curr_thread_angle = std::abs(1.57079632679 - std::asin(Juv.inverse().colwise().normalized().determinant()));
		face_dist(f) = curr_thread_angle;
		if (face_dist(f) > max_dist) {
			max_dist = face_dist(f);
		}
		if (face_dist(f) < min_dist) {
			min_dist = face_dist(f);
		}
	}
	face_dist.array() -= min_dist;
	face_dist /= (max_dist - min_dist); */
}

//main algorithms here
void computeParameterization(int type)
{
	VectorXi fixed_UV_indices;
	MatrixXd fixed_UV_positions;
	SparseMatrix<double> A;
	VectorXd b;
	// Find fixed vertices
	
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
		VS(0) = anchor_idx;

		if (warp_idxs.size() == 0) {//find vertex on boundary as second needed constraint
			VT = boundary;
			igl::exact_geodesic(V, F, VS, FS, VT, FT, Doi);
			for (int j = 0; j < Doi.size(); j++) {//choose the one that is most distant
				if (Doi(j) > max_dist) {
					v2 = VT(j);
					max_dist = Doi(j);
				}
			}
		}
		else {//use one of the already constrained ones
			VT.resize(warp_idxs.size());
			for (int i = 0; i < warp_idxs.size(); i++) {
				VT(i) = warp_idxs[i];
			}
			igl::exact_geodesic(V, F, VS, FS, VT, FT, Doi);
			for (int j = 0; j < Doi.size(); j++) {
				if (Doi(j) > max_dist) {
					v2 = VT(j);
					max_dist = Doi(j);
				}
			}
		}
		computed_free = true;
	}
	if (type == '6' || type == '4') {//for arap and cheby only constrain single point, all possible freedom
		fixed_UV_indices.resize(1);
		fixed_UV_indices(0) = anchor_idx;
		fixed_UV_positions.resize(1, 2);
		fixed_UV_positions << 0, 0;
	}
	else {//need two constraints
		fixed_UV_indices.resize(2);
		fixed_UV_indices(0) = anchor_idx;
		fixed_UV_indices(1) = v2;
		fixed_UV_positions.resize(2, 2);
		fixed_UV_positions << 0, 0, max_dist, 0;
		first_solve = true;
		first_arap_solve=true;
	}

	if (first_arap_solve && first_solve) {//only if not done before, which is case for initialization (lscm) or first iteration of either arap or cheby because both have the same constraints
		ConvertConstraintsToMatrixForm(fixed_UV_indices, fixed_UV_positions, type, C, d);
	}

	// linear system for the parameterization
	A.resize(2 * V.rows(), 2 * V.rows());
	int con = fixed_UV_indices.size();
	if (type == '3') {//LSCM
		computeSurfaceGradientMatrix(Dx, Dy);//results are #Fx#V
		vector<vector<SparseMatrix<double>>> matrices;
		matrices.resize(2);
		matrices[0].push_back(Dx);
		matrices[0].push_back(-Dy);
		matrices[1].push_back(Dy);
		matrices[1].push_back(Dx);
		igl::cat(matrices, A);
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
		b.resize(UV.rows() * 2);
		b.setZero();
		//set sizes for Chebyshev parameterization
		Vs.resize(F.rows());
		Us.resize(F.rows());
		Ss.resize(F.rows());
	}

	if (type == '4') {//ARAP
		SparseMatrix<double> J;
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
		VectorXd R(4 * F.rows());
		for (int f = 0; f < F.rows(); f++) {
			Eigen::Matrix2d U, S, Vf, Juv;
			Juv << Ju(f), Ju(f + F.rows()), Jv(f), Jv(f + F.rows());
			//do local step (rotation fitting)
			SSVD2x2(Juv, U, S, Vf);
			Matrix2d Ri = U * Vf.transpose();
			if (Ri.determinant() < 0) {//no flipping
				Ri = U * flip * Vf.transpose();
			}
			R(f) = Ri(0, 0);
			R(f + F.rows()) = Ri(0, 1);
			R(f + 2 * F.rows()) = Ri(1, 0);
			R(f + 3 * F.rows()) = Ri(1, 1);
		}
		if (first_arap_solve) {
			//build big gradient matrix
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
			//build big mass matrix
			SparseMatrix<double> Mass;
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
			//system matrix
			rhs_mult = A.transpose() * Mass;
			A = (A.transpose() * Mass * A).eval();
			sys_mat = A;
		}
		b = rhs_mult * R;
		A = sys_mat;
	}

	if (type == '6') {//Chebyshev
		SparseMatrix<double> J;


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
		Matrix2d rot, rot2;
		double angle = PI / 16;
		rot << cos(angle), -sin(angle), sin(angle), cos(angle);
		rot2 << cos(PI / 2 - angle), -sin(PI / 2 - angle), sin(PI / 2 - angle), cos(PI / 2 - angle);
		VectorXd R(4 * F.rows());
		double new_global_energy = 0;

		double alt_en = 0;
		double max_per_face = -1;

		for (int f = 0; f < F.rows(); f++) {
			Eigen::Matrix2d Juv;
			Juv << Ju(f), Ju(f + F.rows()), Jv(f), Jv(f + F.rows());
			Eigen::Matrix2d U, S, Vf, Ri;
			MatrixXd I = Matrix2d::Identity();

			if (first_solve) {//initialize
				Matrix2d fake_inverse;
				fake_inverse << Juv(1, 1), -Juv(0, 1), -Juv(1, 0), Juv(0, 0);
				SSVD2x2(((fake_inverse + 0.00001 * I).colwise().normalized()).inverse(), U, S, Vf);
				Us[f] = U;
				Vs[f] = Vf;
				Ss[f] = S;
			}
			else {//take values from previous iteration
				U = Us[f];
				Vf = Vs[f];
				S = Ss[f];
			}
			//based on this do sigma fitting now
			//entries to approximate, a>b
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

			//handle potential shearing limit imposed on local step
			double cos_thread_angle = 0.5 * ((1 / sig) * (1 / sig) - (1 - sig2) * (1 / sig2));
			double cos_thread_angle_limit = cos(thread_angle_limit / 180 * 3.14152);
			if (abs(cos_thread_angle) > abs(cos_thread_angle_limit)) {//limit reached --> project to maximum possible (closest) values instead
				double limited1 = sqrt(1 + cos_thread_angle_limit);
				double limited2 = sqrt(1 - cos_thread_angle_limit);
				if (abs(limited1 - 1 / sig) + abs(limited2 - 1 / sig2) < abs(limited2 - 1 / sig) + abs(limited1 - 1 / sig2)) {
					sig = 1 / limited1;
					sig2 = 1 / limited2;
				}
				else {
					sig = 1 / limited2;
					sig2 = 1 / limited1;
				}
			}

			Matrix2d sigma;
			sigma << sig, 0, 0, sig2;//set sigma matrix accordingly

			//for convergence guarantees: check the energy is not increasing (since we use an approximation it is not guaranteed in any case)
			if ((Juv - U * Ss[f] * Vf.transpose()).squaredNorm() < (Juv - U * sigma * Vf.transpose()).squaredNorm()) {
				sigma = Ss[f];//keep old one
			}

			per_face_energy(f) = (Juv - U * sigma * Vf.transpose()).squaredNorm();

			//fit V (rotation matrix) using Procrustes
			Matrix2d V_fit, S_svd, U_svd, V_svd;
			SSVD2x2(sigma * U.transpose() * Juv, U_svd, S_svd, V_svd);
			V_fit = V_svd * U_svd.transpose();
			Eigen::Matrix2d flip = Eigen::Matrix2d::Identity();
			flip(1, 1) = -1.;
			if (V_fit.determinant() < 0) {//no flipping
				V_fit = V_svd * flip * U_svd.transpose();
			}
			//update matrices
			Vs[f] = V_fit;
			Ss[f] = sigma;
			Us[f] = U;
			//construct Cheby Jacobian
			Ri = U * sigma * V_fit.transpose();			

			//update energies, maximal cheby error value etc
			per_face_energy(f) = area(f) * (Juv - Ri).squaredNorm();//our chebyshev energy
			if ((Juv.inverse() - Juv.inverse().colwise().normalized()).norm() > max_per_face) {
				max_per_face = (Juv.inverse() - Juv.inverse().colwise().normalized()).norm();
			}
			new_global_energy += per_face_energy(f);
			alt_en += area(f) * (Juv.inverse() - Juv.inverse().colwise().normalized()).squaredNorm();//cheby error as defined in paper
			//fill in vector
			R(f) = Ri(0, 0);
			R(f + F.rows()) = Ri(0, 1);
			R(f + 2 * F.rows()) = Ri(1, 0);
			R(f + 3 * F.rows()) = Ri(1, 1);
		}
		if (new_global_energy > global_energy + 0.000001 && !first_solve) {//should not happen
			cout << "ATTENTION: global energy is increasing by " << (new_global_energy - global_energy) / global_energy << endl;
		}
		//save energies 
		ad_energy_coll.push_back(alt_en);
		lg_energy_coll.push_back(new_global_energy);
		global_energy = new_global_energy;
		if (first_solve) {
			//build big gradient matrix
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
			//build big mass matrix
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
			//system
			rhs_mult = A.transpose() * Mass;
			sys_mat = (A.transpose() * Mass * A).eval();
		}
		b = rhs_mult * R;
		A = sys_mat;
	}


	// Solve the linear system.
	int num_con = 2 * fixed_UV_indices.size() + warp_idxs.size() + weft_idxs.size();
	if (type == '3' && warp_idxs.size() != 0) {//one warp index was used as fully fixed
		num_con--;
	}
	if (type == '3' || (type == '6' && first_solve) || (type == '4' && first_arap_solve)) {//factorize the system (unless we're in the middle of cheby or arap iterations)
		SparseMatrix<double> sys;
		SparseMatrix<double> z = SparseMatrix<double>(num_con, num_con);
		z.setZero();
		vector<vector<SparseMatrix<double>>> matrices;
		matrices.resize(2);
		//stack system matrix defined in method with constraints to solve in a least squares manner
		matrices[0].push_back(A);
		matrices[0].push_back(C.transpose());
		matrices[1].push_back(C);
		matrices[1].push_back(z);
		igl::cat(matrices, sys);
		solver.analyzePattern(sys);
		solver.factorize(sys);
		//cout << solver.lastErrorMessage() << endl;//in case there is singularity issue, let user know
		//cout << solver.info() << endl;
		//flag that it has already been factorized
		if (type == '6') {
			first_solve = false;
		}
		if (type == '4') {
			first_arap_solve = false;
		}
	}
	//solve system
	VectorXd rhs;
	rhs.resize(2 * UV.rows() + num_con);
	rhs << b, d;
	VectorXd sol = solver.solve(rhs);
	UV.col(0) = sol.block(0, 0, UV.rows(), 1);
	UV.col(1) = sol.block(UV.rows(), 0, UV.rows(), 1);
	if (first_solve && type == '3') {//save initialization for other uses
		MatrixXd param_3d;
		param_3d.resize(UV.rows(), 3);
		param_3d.col(0) = UV.col(0);
		param_3d.col(1) = UV.col(1);
		param_3d.col(2).setZero();
		//igl::writeOFF("./lscm_init_param.off", param_3d, Ftex);
	}
	if (type != '6') {//again saved for other uses
		init.resize(UV.rows(), 2);
		init.col(0) = UV.col(0);
		init.col(1) = UV.col(1);
	}
}

bool callback_key_pressed(Viewer& viewer, unsigned char key, int modifiers) {
	switch (key) {
	case '3':
		computeParameterization(key);
		break;
	case '0':
		TextureResolution /= 2;
		break;
	case '-':
		TextureResolution *= 2;
		break;
	case ' ': // space bar -  switches view between mesh and parameterization
		if (showingUV)
		{
			temp2D = viewer.core();
			viewer.core() = temp3D;
			showingUV = false;
		}
		else
		{
			if (UV.rows() > 0)
			{
				temp3D = viewer.core();
				viewer.core() = temp2D;
				showingUV = true;
			}
			else { std::cout << "ERROR ! No valid parameterization\n"; }
		}
		break;
	}
	if (key == '6' || key == '4') {
		vector<double> runtimes;
		vector<double> runtimes_first;

		for (int z = 0; z < averaging_runtime_iters; z++) {
			UV = init;
			converged = false;
			first_solve = true;
			first_arap_solve = true;
			ad_energy_coll.clear();
			lg_energy_coll.clear();
			//save energy of initialization too
			VectorXd Ju, Jv;
			vector<vector<SparseMatrix<double>>> mat_vec;
			mat_vec.resize(2);
			mat_vec[0].push_back(Dx);
			mat_vec[1].push_back(Dy);
			SparseMatrix<double> J;
			igl::cat(mat_vec, J);
			Ju = J * UV.col(0);
			Jv = J * UV.col(1);
			double alt_en = 0;//compute cheby error
			for (int f = 0; f < F.rows(); f++) {
				Eigen::Matrix2d Juv;
				Juv << Ju(f), Ju(f + F.rows()), Jv(f), Jv(f + F.rows());
				alt_en += area(f) * (Juv.inverse() - Juv.inverse().colwise().normalized()).squaredNorm();
			}
			ad_energy_coll.push_back(alt_en);
			std::cout << std::fixed << std::setprecision(9) << std::left;
			int i = 1;
			const auto start = std::chrono::high_resolution_clock::now();
			computeParameterization(key);
			MatrixXd oldUV = UV;
			const auto end_first = std::chrono::high_resolution_clock::now();//time first iteration (includes factorization)
			while (!converged && i < max_iters) {//do the rest of the iterations until convergence
				computeParameterization(key);
				i++;
				if ((oldUV / oldUV.norm() - UV / oldUV.norm()).norm() < convergence_precision) {
					converged = true;
				}
				oldUV = UV;
			}

			const auto end = std::chrono::high_resolution_clock::now();
			const std::chrono::duration<double> diff_c = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
			runtimes.push_back(diff_c.count());
			num_iters_until_convergence = i;
			const std::chrono::duration<double> diff_first = std::chrono::duration_cast<std::chrono::duration<double>>(end_first - start);
			runtimes_first.push_back(diff_first.count());
		}
		double max_runtime = -1;
		double min_runtime = 10000000000;
		runtime = 0;
		for (int z = 0; z < averaging_runtime_iters; z++) {
			runtime += runtimes[z];
			if (runtimes[z] > max_runtime) {
				max_runtime = runtimes[z];
			}
			if (runtimes[z] < min_runtime) {
				min_runtime = runtimes[z];
			}
		}
		if(averaging_runtime_iters>2){
			//filter out outliers
			runtime -= min_runtime;
			runtime -= max_runtime;
			//average
			runtime /= (averaging_runtime_iters - 2);
		}
		else{
			runtime/= averaging_runtime_iters;
		}

		//same for first
		max_runtime = -1;
		min_runtime = 10000000000;
		runtime_first = 0;
		for (int z = 0; z < averaging_runtime_iters; z++) {
			runtime_first += runtimes_first[z];
			if (runtimes_first[z] > max_runtime) {
				max_runtime = runtimes_first[z];
			}
			if (runtimes_first[z] < min_runtime) {
				min_runtime = runtimes_first[z];
			}
		}
		if(averaging_runtime_iters>2){
			//filter out outliers
			runtime_first -= min_runtime;
			runtime_first -= max_runtime;
			//average
			runtime_first /= (averaging_runtime_iters - 2);
		}
		else{
			runtime_first/= averaging_runtime_iters;
		}

	}
	Redraw();
	if (key == 'v') {
		VectorXd distortion;
		cheby_distortion(distortion);
		distortion *= (1.0 / distortion.maxCoeff());
		MatrixXd face_col(F.rows(), 3);
		face_col.col(1) = Eigen::VectorXd::Ones(F.rows()) - distortion;
		face_col.col(2) = Eigen::VectorXd::Ones(F.rows()) - distortion;
		face_col.col(0).setOnes();
		viewer.data().show_texture = false;
		viewer.data().set_colors(face_col);
	}
	return true;
}

int main(int argc, char* argv[]) {
	if (argc < 2) {
		cout << "Usage ex4_bin <mesh.off/obj>" << endl;
		load_mesh("../data/hemisphere.obj");
	}
	else
	{
		// Read points and normals
		igl::read_triangle_mesh("../data/"+std::string(argv[1]), V, F);
		VectorXi comp;
		igl::vertex_components(F, comp);//compute pieces
		int num_comp=comp.maxCoeff()+1;
		if (num_comp > 1) {//disjoint, given multiple components
			cout<<"WARNING: disjoint mesh"<<endl;
		}
	}
	//set up viewer
	Redraw();
	viewer.core().align_camera_center(V);
	showingUV = false;
	anchor_col.resize(1, 3);//define constraint colors
	anchor_col << 0.9, 0.3, 0.3;
	warp_col.resize(1, 3);
	warp_col << 0.3, 0.9, 0.3;
	weft_col.resize(1, 3);
	weft_col << 0.3, 0.3, 0.9;
	//initialize variables
	per_face_energy.resize(F.rows());
	per_face_energy.fill(INFINITY);
	global_energy = INFINITY;
	igl::doublearea(V, F, area);
	area *= 0.5;
	M=area.asDiagonal();//igl::diag(area, M);
	flip(1, 1) = -1.;
	Ftex = F;
	UV.resize(V.rows(), 2);
	UV.setZero();
	anchor_idx = 0;
    //add menu
    igl::opengl::glfw::imgui::ImGuiPlugin imgui_plugin;
    viewer.plugins.push_back(&imgui_plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    imgui_plugin.widgets.push_back(&menu);
	menu.callback_draw_viewer_menu = [&]()
	{
		// Add new group
		if (ImGui::CollapsingHeader("Set Constraints", ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (ImGui::Button("Set Anchor", ImVec2(-1, 0)))
			{
				fix_weft = false;
				weft_mode = "Constrain weft direction";
				fix_warp = false;
				warp_mode = "Constrain warp direction";
				fix_anchor = true;
			}
			if (ImGui::Button(warp_mode, ImVec2(-1, 0)))
			{
				if (fix_warp) {
					fix_warp = false;
					warp_mode = "Constrain warp direction";
				}
				else {
					fix_warp = true;
					warp_mode = "End fixing warp direction";
				}
				fix_weft = false;
				fix_anchor = false;
				weft_mode = "Constrain weft direction";;
			}
			if (ImGui::Button(weft_mode, ImVec2(-1, 0)))
			{
				if (fix_weft) {
					fix_weft = false;
					weft_mode = "Constrain weft direction";
				}
				else {
					fix_weft = true;
					weft_mode = "End fixing weft direction";
				}
				fix_warp = false;
				fix_anchor = false;
				warp_mode = "Constrain warp direction";
			}
		}
		if (ImGui::InputDouble("Texture resolution", &TextureResolution, 0, 0)) {
			Redraw();
		}
		if (ImGui::InputDouble("Thread angle limit [0,90]", &thread_angle_limit, 0, 0));
		if (ImGui::Button("load texture from .obj", ImVec2(-1, 0)))
		{
			//for pre-parametrized meshes
			MatrixXd VT, CN, FN;
			MatrixXi FT;
			igl::readOBJ(argv[1], V, VT, CN, F, FT, FN);
			UV = VT;
			Ftex = FT;
			first_solve = true;

			viewer.data().clear();
			VectorXi comp;
			igl::vertex_components(FT, comp);//compute pieces
			int num_comp=comp.maxCoeff()+1;
			if(num_comp>1){
				cout<<"WARNING: disjoint mesh"<<endl;
			}
			vector<int> uv_to_v_corr;
			uv_to_v_corr.resize(UV.rows());
			for (int f = 0; f < Ftex.rows(); f++) {
				uv_to_v_corr[Ftex(f, 0)] = F(f, 0);
				uv_to_v_corr[Ftex(f, 1)] = F(f, 1);
				uv_to_v_corr[Ftex(f, 2)] = F(f, 2);
			}
			init = VT;
			cout << "loaded texture from file" << endl;
		}
		if (ImGui::CollapsingHeader("Saving and displaying information", ImGuiTreeNodeFlags_DefaultOpen))
		{
			ImGui::InputText("File name", save_name);
			if (ImGui::Button("save .obj and stats", ImVec2(-1, 0)))
			{
				//save textured mesh
				std::fstream s{ "../res/" + save_name + ".obj", s.binary | s.trunc | s.in | s.out };
				for (int i = 0; i < V.rows(); i++) {
					s << "v " << V(i, 0) << " " << V(i, 1) << " " << V(i, 2) << std::endl;
				}
				for (int i = 0; i < UV.rows(); i++) {
					s << "vt " << UV(i, 0) << " " << UV(i, 1) << std::endl;
				}
				for (int i = 0; i < F.rows(); i++) {
					s << "f " << F(i, 0) + 1 << "/" << F(i, 0) + 1 << " "
						<< F(i, 1) + 1 << "/" << F(i, 1) + 1 << " "
						<< F(i, 2) + 1 << "/" << F(i, 2) + 1 << " " << std::endl;
				}
				s.close();
				//save metrics in .txt
				std::ofstream outFile("../res/" + save_name + ".txt");
				// Check if the file opened successfully
				if (!outFile) {
					std::cerr << "Failed to open the file for writing!" << std::endl;
				}
				// Write some text to the file
				outFile << "metrics for " << save_name << std::endl;
				//save energies (convergence etc)
				outFile << "parameterization energies (difference between j.inv and the closest cheby jacobian) are" << endl;
				for (int i = 0; i < ad_energy_coll.size(); i++) {
					outFile << ad_energy_coll[i] << ", ";
				}
				outFile << endl << "flattening energies (our defined local-global energy) are" << endl;
				for (int i = 0; i < lg_energy_coll.size(); i++) {
					outFile << lg_energy_coll[i] << ", ";
				}
				outFile << endl;
				//compute metrics for whatever current configuration (this should also work for parameterizations loaded from other methods etc)
				//compute jacobian
				VectorXd Ju, Jv;
				computeSurfaceGradientMatrix(Dx, Dy);//results are #Fx#V
				vector<vector<SparseMatrix<double>>> mat_vec;
				mat_vec.resize(2);
				mat_vec[0].push_back(Dx);
				mat_vec[1].push_back(Dy);
				SparseMatrix<double> J;
				igl::cat(mat_vec, J);
				Ju = J * UV.col(0);
				Jv = J * UV.col(1);
				double max_cheby = -1;
				double avg_cheby = 0;
				double min_cheby = 100;
				double total_area = 0;
				outFile << "per face cheby distortion: (cheby distortion which is inverse-normalized inverse)" << endl;
				for (int f = 0; f < F.rows(); f++) {
					Eigen::Matrix2d Juv;
					Juv << Ju(f), Ju(f + F.rows()), Jv(f), Jv(f + F.rows());
					double curr_cheby_distortion = (Juv.inverse() - Juv.inverse().colwise().normalized()).squaredNorm();
					outFile << curr_cheby_distortion << endl;
					if (curr_cheby_distortion > max_cheby) {
						max_cheby = curr_cheby_distortion;
					}
					if (curr_cheby_distortion < min_cheby) {
						min_cheby = curr_cheby_distortion;
					}
					avg_cheby += area(f) * curr_cheby_distortion;
					total_area += area(f);

				}
				outFile << "max cheby distortion: " << max_cheby << endl;
				outFile << "min cheby distortion: " << min_cheby << endl;
				outFile << "avg (energy divided by area) cheby distortion: " << avg_cheby / total_area << endl;
				outFile << "energy cheby distortion: " << avg_cheby << endl;

				outFile << "per face shearing" << endl;
				for (int f = 0; f < F.rows(); f++) {
					Eigen::Matrix2d Juv;
					Juv << Ju(f), Ju(f + F.rows()), Jv(f), Jv(f + F.rows());
					double curr_thread_angle = std::abs(1.57079632679 - std::asin(Juv.inverse().colwise().normalized().determinant()));
					outFile << curr_thread_angle << endl;
				}

				outFile << "runtime for " << num_iters_until_convergence << " iterations (until convergence, meaning energy change <" << convergence_precision << ", or reaching threshold of " << max_iters << "): " << runtime << " s" << endl;
				outFile << "of that runtime,  " << runtime_first << " s were needed for the first iteration (including factorization of system)" << endl;
				outFile << "mesh has " << V.rows() << " vertices and " << F.rows() << " faces" << endl;

				// Close the file
				outFile.close();
				std::cout << "Mesh and metrics have been saved!" << std::endl;

			}
			if (ImGui::Button("hide pts", ImVec2(-1, 0)))
			{
				viewer.data().clear_points();
			}
			if (ImGui::Button("print energies", ImVec2(-1, 0)))
		{
			cout << "parameterization energies (difference between j.inv and the closest cheby jacobian) are" << endl;
			for (int i = 0; i < ad_energy_coll.size(); i++) {
				cout << ad_energy_coll[i] << ", ";
			}
			cout << endl << endl << "flattening energies (our defined local-global energy) are" << endl;
			for (int i = 0; i < ad_energy_coll.size(); i++) {
				cout << lg_energy_coll[i] << ", ";
			}
			cout << endl << endl;
		}
		}
	};

	viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer& viewer, int, int)->bool
	{
		if (fix_warp || fix_weft || fix_anchor) {
			int fid;
			Eigen::Vector3f bc;
			// Cast a ray in the view direction starting from the mouse position
			double x = viewer.current_mouse_x;
			double y = viewer.core().viewport(3) - viewer.current_mouse_y;
			MatrixXd unproj_v;
			MatrixXi unproj_f;
			if (showingUV) {
				unproj_f = Ftex;
				unproj_v = UV;
				cout << "uv used for fixing" << endl;
			}
			else {
				unproj_f = F;
				unproj_v = V;
			}
			cout << "before unproj" << endl;
			if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core().view,
				viewer.core().proj, viewer.core().viewport, unproj_v, unproj_f, fid, bc))
			{
				cout << "wahoo" << endl;
				float max = bc.maxCoeff();//point that is closest
				int point_face_idx = 0;//find argmax
				for (int i = 0; i < 3; i++) {
					if (bc(i) == max) {
						point_face_idx = i;
					}
				}
				int point = F(fid, point_face_idx);//indexes into V
				MatrixXd col;
				cout << point << endl;
				if (fix_warp) {
					col = warp_col;
					warp_idxs.push_back(point);
					cout << "warp" << point << endl;
				}
				if (fix_weft) {
					col = weft_col;
					weft_idxs.push_back(point);
				}
				if (fix_anchor) {
					col = anchor_col;
					fix_anchor = false;
					anchor_idx = point;
				}
				viewer.data().add_points(unproj_v.row(point), col);
				return true;
			}
			return false;
		}
		return false;
	};

	viewer.callback_key_pressed = callback_key_pressed;
	viewer.callback_mouse_move = callback_mouse_move;
	viewer.callback_init = callback_init;
	viewer.core().background_color.setOnes();
	viewer.data().show_lines = false;
    viewer.data().set_colors(mesh_color);
	viewer.launch();
}