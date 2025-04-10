#ifndef GEODESY_H
#define GEODESY_H

#include <cmath>
#include <stdexcept>

/*!
 * @brief 坐标变换
 */
namespace geodesy {

/*!
 * @brief 参考椭球
 *
 */
enum class refEllipsoid { wgs84, grs80 };

namespace ellipsoid {
/*!
 * @brief               计算参考椭球长轴
 * @param               参考椭球模型
 * @return              参考椭球的长轴，单位 m
 * @note                长轴指参考椭球的最长直径
 */
inline constexpr double getMajor(refEllipsoid ref_ellipsoid) {
  switch (ref_ellipsoid) {
  case geodesy::refEllipsoid::wgs84:
    return 6378137.0;
  case geodesy::refEllipsoid::grs80:
    return 6378137.0;
  default:
    return getMajor(geodesy::refEllipsoid::wgs84);
  }
}

/*!
 * @brief               计算参考椭球短轴
 * @param               参考椭球模型
 * @return              参考椭球的短轴，单位 m
 * @note                短轴指参考椭球的最短直径
 */
inline constexpr double getMinor(refEllipsoid ref_ellipsoid) {
  switch (ref_ellipsoid) {
  case geodesy::refEllipsoid::wgs84:
    return 6356752.31424518;
  case geodesy::refEllipsoid::grs80:
    return 6356752.31414036;
  default:
    return getMinor(geodesy::refEllipsoid::wgs84);
  }
}

/*!
 * @brief               计算参考椭球极偏率
 * @param               参考椭球模型
 * @return              参考椭球的极偏率
 * @note                极偏率指椭球相对完美球体的偏度
 */
inline constexpr double getFlattening(refEllipsoid ref_ellipsoid) {
  double major = getMajor(ref_ellipsoid);
  double minor = getMinor(ref_ellipsoid);
  return (major - minor) / major;
}

/*!
 * @brief               计算参考椭球的平方离心率
 * @param               参考椭球模型
 * @return              参考椭球的平方离心率
 * @note                平方离心率是极偏率的对偶形式
 */
inline constexpr double getFirstEcc2(refEllipsoid ref_ellipsoid) {
  double major = getMajor(ref_ellipsoid);
  double f = getFlattening(ref_ellipsoid);
  return f * (2 - f);
}
} // namespace ellipsoid

/*!
 * @brief         贴面坐标系，用于计算偏移量
 *
 * @param         east
 * @param         north
 * @param         up
 * @param         lat
 * @param         lon
 * @param         u
 * @param         v
 * @param         w
 * @attention
 */
inline void enu2uvw(double east, double north, double up, double lat,
                    double lon, double &u, double &v, double &w) {
  double t = std::cos(lat) * up - std::sin(lat) * north;
  u = std::cos(lon) * t - std::sin(lon) * east;
  v = std::sin(lon) * t + std::cos(lon) * east;
  w = std::sin(lat) * up + std::cos(lat) * north;
}

/*!
 * @brief         贴面坐标系，用于计算偏移量
 *
 * @param         u
 * @param         v
 * @param         w
 * @param         lat
 * @param         lon
 * @param         east
 * @param         north
 * @param         up
 * @attention
 */
inline void uvw2enu(double u, double v, double w, double lat, double lon,
                    double &east, double &north, double &up) {
  double t = std::cos(lon) * u + std::sin(lon) * v;
  east = -1 * std::sin(lon) * u + std::cos(lon) * v;
  north = -1 * std::sin(lat) * t + std::cos(lat) * w;
  up = std::cos(lat) * t + std::sin(lat) * w;
}

/*!
 * @brief 坐标变换：ECEF ==> Geodetic
 *
 * @param x             目标点 x 坐标，单位 m
 * @param y             目标点 y 坐标，单位 m
 * @param z             目标点 z 坐标，单位 m
 * @param[out] lat      目标点纬度，单位 rad
 * @param[out] lon      目标点经度，单位 rad
 * @param[out] alt      目标点高度，单位 m
 * @param               参考椭球模型，'wgs84'（默认） 或 'grs80'
 * @ref You, Rey-Jer. (2000). Transformation of Cartesian to Geodetic
 * Coordinates without Iterations. Journal of Surveying Engineering.
 * doi: 10.1061/(ASCE)0733-9453.
 *
 */
inline void ecef2geodetic(double x, double y, double z, double &lat,
                          double &lon, double &alt,
                          refEllipsoid ref_ellipsoid = refEllipsoid::wgs84) {
  double major = ellipsoid::getMajor(ref_ellipsoid);
  double minor = ellipsoid::getMinor(ref_ellipsoid);
  double e = std::sqrt(ellipsoid::getMinor(ref_ellipsoid));

  double r = std::sqrt(x * x + y * y + z * z);
  double var = r * r - e * e;
  double u =
      std::sqrt(0.5 * var + 0.5 * std::sqrt(var * var + 4.0 * e * e * z * z));

  double q = std::sqrt(x * x + y * y);
  double hu_e = std::sqrt(u * u + e * e);

  // 可能为 nan
  double beta = std::atan(hu_e / u * z / q);

  if (isnan(beta)) {
    if (std::abs(z) < 1.0e-9) {
      beta = 0;
    } else if (z > 0) {
      beta = M_PI;
    } else {
      beta = -M_PI;
    }
  }

  double eps = ((minor * u - major * hu_e + e * e) * std::sin(beta)) /
               (major * hu_e / std::cos(beta) - e * e * std::cos(beta));

  beta += eps;

  lat = std::atan(major / minor * std::tan(beta));
  lon = std::atan2(y, x);

  double v1 = z - minor * std::sin(beta);
  double v2 = q - major * std::cos(beta);

  if ((x * x / major / major) + (y * y / major / major) +
          (z * z / minor / minor) <
      1.0) {
    alt = -1 * std::sqrt(v1 * v1 + v2 * v2);
  } else {
    alt = std::sqrt(v1 * v1 + v2 * v2);
  }
}
/*!
 * @brief 坐标变换：AER ==> ENU
 *
 * @param az            目标点方位角，单位 rad
 * @param elev          目标点俯仰角，单位 rad
 * @param range         目标点斜距，单位 m
 * @param[out] east     目标点东坐标，单位 m
 * @param[out] north    目标点北坐标，单位 m
 * @param[out] up       目标点天坐标，单位 m
 *
 * @throws std::domain_error if range < 0
 */
inline void aer2enu(double az, double elev, double range, double &east,
                    double &north, double &up) {
  if (range < 0) {
    throw std::domain_error("range should not be negative");
  }

  auto r = range * std::cos(elev);
  east = r * std::sin(az);
  north = r * std::cos(az);
  up = range * std::sin(elev);
}
/*!
 * @brief 坐标变换：ENU ==> AER
 *
 * @param east          目标点东坐标，单位 m
 * @param north         目标点北坐标，单位 m
 * @param up            目标点上坐标，单位 m
 * @param[out] az       目标点方位角，单位 rad
 * @param[out] elev     目标点俯仰角，单位 rad
 * @param[out] range    目标点斜距，单位 m
 *
 */
inline void enu2aer(double east, double north, double up, double &az,
                    double &elev, double &range) {
  // 1 millimeter precision for singularity stability
  if (std::abs(east) < 1.0e-3) {
    east = 0;
  }
  if (std::abs(north) < 1.0e-3) {
    north = 0;
  }
  if (std::abs(up) < 1.0e-3) {
    up = 0;
  }

  double r = hypot(east, north);

  range = hypot(r, up);
  elev = std::atan2(up, r);

  az = std::fmod(2 * M_PI + std::fmod(std::atan2(east, north), 2 * M_PI),
                 2 * M_PI);
}

/*!
 * @brief 坐标变换：AER ==> NED
 *
 * @param az            目标点方位角，单位 rad
 * @param elev          目标点俯仰角，单位 rad
 * @param range         目标点斜距，单位 m
 * @param[out] north    目标点北坐标，单位 m
 * @param[out] east     目标点东坐标，单位 m
 * @param[out] down     目标点下坐标，单位 m
 *
 */
inline void aer2ned(double az, double el, double range, double &north,
                    double &east, double &down) {
  aer2enu(az, el, range, east, north, down);
  down = down * -1;
}

/*!
 * @brief 坐标变换：Geodetic ==> ECEF
 *
 * @param lat           目标点纬度，单位 rad
 * @param lon           目标点经度，单位 rad
 * @param alt           目标点高度，单位 m
 * @param[out] x        目标点 x 坐标，单位 m.
 * @param[out] y        目标点 y 坐标，单位 m.
 * @param[out] z        目标点 z 坐标，单位 m.
 * @param               参考椭球模型，'wgs84'（默认） 或 'grs80'
 *
 * @throws std::domain_error if latitude is not within [-π/2, π/2].
 */
inline void geodetic2ecef(double lat, double lon, double alt, double &x,
                          double &y, double &z,
                          refEllipsoid ref_ellipsoid = refEllipsoid::wgs84) {

  if (std::abs(lat) > M_PI / 2) {
    throw std::domain_error("-pi/2 <= latitude <= pi/2");
  }

  double major = ellipsoid::getMajor(ref_ellipsoid);
  double minor = ellipsoid::getMinor(ref_ellipsoid);
  double e2 = ellipsoid::getFirstEcc2(ref_ellipsoid);

  double nr = major / std::sqrt(((1.0 - e2 * std::sin(lat) * std::sin(lat))));

  x = (nr + alt) * std::cos(lat) * std::cos(lon);
  y = (nr + alt) * std::cos(lat) * std::sin(lon);
  z = (nr * (minor / major) * (minor / major) + alt) * std::sin(lat);
}

/*!
 * @brief 坐标变换：ECEF ==> ENU
 *
 * @param x             目标点 x 坐标，单位 m
 * @param y             目标点 y 坐标，单位 m
 * @param z             目标点 z 坐标，单位 m
 * @param lat0          参考点纬度，单位 rad
 * @param lon0          参考点经度，单位 rad
 * @param alt0          参考点高度，单位 m
 * @param[out] east     目标点东坐标，单位 m
 * @param[out] north    目标点北坐标，单位 m
 * @param[out] up       目标点上坐标，单位 m
 * @param               参考椭球模型，'wgs84'（默认） 或 'grs80'
 *
 */
inline void ecef2enu(double x, double y, double z, double lat0, double lon0,
                     double alt0, double &east, double &north, double &up,
                     refEllipsoid ref_ellipsoid = refEllipsoid::wgs84) {
  double x0, y0, z0;
  geodetic2ecef(lat0, lon0, alt0, x0, y0, z0, ref_ellipsoid);
  double dx, dy, dz;
  dx = x - x0;
  dy = y - y0;
  dz = z - z0;
  uvw2enu(dx, dy, dz, lat0, lon0, east, north, up);
}

/*!
 * @brief 坐标变换：ECEF ==> AER
 *
 * @param x           目标点 x 坐标，单位 m
 * @param y           目标点 y 坐标，单位 m
 * @param z           目标点 z 坐标，单位 m
 * @param lat         目标点纬度，单位 rad
 * @param lon         目标点经度，单位 rad
 * @param alt         目标点高度，单位 m
 * @param[out] az     目标点方位角，单位 rad
 * @param[out] elev   目标点俯仰角，单位 rad
 * @param[out] range  目标点斜距，单位 m
 * @param             参考椭球模型，'wgs84'（默认） 或 'grs80'
 *
 */
inline void ecef2aer(double x, double y, double z, double lat, double lon,
                     double alt, double &az, double &el, double &range,
                     refEllipsoid ref_ellipsoid = refEllipsoid::wgs84) {
  double east, north, up;
  ecef2enu(x, y, z, lat, lon, alt, east, north, up, ref_ellipsoid);
  enu2aer(east, north, up, az, el, range);
}

/*!
 * @brief 坐标变换：ECEF ==> NED
 *
 * @param x             目标点x 坐标，单位 m
 * @param y             目标点y 坐标，单位 m
 * @param z             目标点z 坐标，单位 m
 * @param lat0          参考点点纬度，单位 rad
 * @param lon0          参考点点经度，单位 rad
 * @param alt0          参考点点高度，单位 m
 * @param[out] north    目标点北坐标，单位 m
 * @param[out] east     目标点东坐标，单位 m
 * @param[out] down     目标点下坐标，单位 m
 * @param               参考椭球模型，'wgs84'（默认） 或 'grs80'
 *
 */
inline void ecef2ned(double x, double y, double z, double lat0, double lon0,
                     double alt0, double &north, double &east, double &down,
                     refEllipsoid ref_ellipsoid = refEllipsoid::wgs84) {
  ecef2enu(x, y, z, lat0, lon0, alt0, east, north, down, ref_ellipsoid);
  down = down * -1;
}

/*!
 * @brief 坐标变换：ENU ==> ECEF
 *
 * @param east        目标点东坐标，单位 m
 * @param north       目标点北坐标，单位 m
 * @param up          目标点上坐标，单位 m
 * @param lat0        参考点纬度，单位 rad
 * @param lon0        参考点经度，单位 rad
 * @param alt0        参考点高度，单位 m
 * @param[out] x      目标点 x 坐标，单位 m
 * @param[out] y      目标点 y 坐标，单位 m
 * @param[out] z      目标点 z 坐标，单位 m
 * @param             参考椭球模型，'wgs84'（默认） 或 'grs80'
 *
 */
inline void enu2ecef(double east, double north, double up, double lat0,
                     double lon0, double alt0, double &x, double &y, double &z,
                     refEllipsoid ref_ellipsoid = refEllipsoid::wgs84) {
  double x0, y0, z0;
  double dx, dy, dz;

  geodetic2ecef(lat0, lon0, alt0, x0, y0, z0, ref_ellipsoid);
  enu2uvw(east, north, up, lat0, lon0, dx, dy, dz);
  x = x0 + dx;
  y = y0 + dy;
  z = z0 + dz;
}

/*!
 * @brief 坐标变换：ENU ==> Geodetic
 *
 * @param east        目标点东坐标，单位 m
 * @param north       目标点北坐标，单位 m
 * @param up          目标点上坐标，单位 m
 * @param lat0        参考点纬度，单位 rad
 * @param lon0        参考点经度，单位 rad
 * @param alt0        参考点高度，单位 m
 * @param[out] lat    目标点纬度，单位 rad
 * @param[out] lon    目标点经度，单位 rad
 * @param[out] alt    目标点高度，单位 m
 * @param             参考椭球模型，'wgs84'（默认） 或 'grs80'
 *
 */
inline void enu2geodetic(double east, double north, double up, double lat0,
                         double lon0, double alt0, double &lat, double &lon,
                         double &alt,
                         refEllipsoid ref_ellipsoid = refEllipsoid::wgs84) {
  double x, y, z;
  enu2ecef(east, north, up, lat0, lon0, alt0, x, y, z, ref_ellipsoid);
  ecef2geodetic(x, y, z, lat, lon, alt, ref_ellipsoid);
}

/*!
 * @brief 坐标变换：Geodetic ==> ENU
 *
 * @param lat           目标点纬度，单位 rad
 * @param lon           目标点经度，单位 rad
 * @param alt           目标点高度，单位 m
 * @param lat0          纬度，单位 rad
 * @param lon0          经度，单位 rad
 * @param alt0          高度，单位 m
 * @param[out] east     东坐标，单位 m
 * @param[out] north    北坐标，单位 m
 * @param[out] up       上坐标，单位 m
 * @param               参考椭球模型，'wgs84'（默认） 或 'grs80'
 *
 */

inline void geodetic2enu(double lat, double lon, double alt, double lat0,
                         double lon0, double alt0, double &east, double &north,
                         double &up,
                         refEllipsoid ref_ellipsoid = refEllipsoid::wgs84) {
  double x1, y1, z1;
  geodetic2ecef(lat, lon, alt, x1, y1, z1, ref_ellipsoid);

  double x2, y2, z2;
  geodetic2ecef(lat0, lon0, alt0, x2, y2, z2, ref_ellipsoid);

  uvw2enu(x1 - x2, y1 - y2, z1 - z2, lat0, lon0, east, north, up);
}

/*!
 * @brief 坐标变换：Geodetic ==> AER
 *
 * @param lat           目标点纬度，单位 rad
 * @param lon           目标点经度，单位 rad
 * @param alt           目标点高度，单位 m
 * @param lat0          参考点纬度，单位 rad
 * @param lon0          参考点经度，单位 rad
 * @param alt0          参考点高度，单位 m
 * @param[out] az       目标点方位角，单位 rad
 * @param[out] elev     目标点俯仰角，单位 rad
 * @param[out] range    目标点斜距，单位 m
 * @param               参考椭球模型，'wgs84'（默认） 或 'grs80'
 *
 */
inline void geodetic2aer(double lat, double lon, double alt, double lat0,
                         double lon0, double alt0, double &az, double &el,
                         double &range,
                         refEllipsoid ref_ellipsoid = refEllipsoid::wgs84) {
  double east, north, up;
  geodetic2enu(lat, lon, alt, lat0, lon0, alt0, east, north, up, ref_ellipsoid);
  enu2aer(east, north, up, az, el, range);
}

/*!
 * @brief 坐标变换：Geodetic ==> NED
 *
 * @param lat           目标点纬度，单位 rad
 * @param lon           目标点经度，单位 rad
 * @param alt           目标点高度，单位 m
 * @param lat0          参考点纬度，单位 rad
 * @param lon0          参考点经度，单位 rad
 * @param alt0          参考点高度，单位 m
 * @param[out] north    目标点北坐标，单位 m
 * @param[out] east     目标点东坐标，单位 m
 * @param[out] down     目标点下坐标，单位 m
 * @param               参考椭球模型，'wgs84'（默认） 或 'grs80'
 *
 */
inline void geodetic2ned(double lat, double lon, double alt, double lat0,
                         double lon0, double alt0, double &north, double &east,
                         double &down,
                         refEllipsoid ref_ellipsoid = refEllipsoid::wgs84) {
  geodetic2enu(lat, lon, alt, lat0, lon0, alt0, east, north, down,
               ref_ellipsoid);
  down = down * -1;
}

/*!
 * @brief 坐标变换：AER ==> ECEF
 *
 * @param az            目标点方位角，单位 rad
 * @param elev          目标点俯仰角，单位 rad
 * @param range         目标点斜距，单位 m
 * @param lat0           参考点纬度，单位 rad
 * @param lon0           参考点经度，单位 rad
 * @param alt0           参考点高度，单位 m
 * @param[out] x        目标点 x 坐标，单位 m.
 * @param[out] y        目标点 y 坐标，单位 m.
 * @param[out] z        目标点 z 坐标，单位 m.
 * @param               参考椭球模型，'wgs84'（默认） 或 'grs80'
 *
 */
inline void aer2ecef(double az, double elev, double range, double lat0,
                     double lon0, double alt0, double &x, double &y, double &z,
                     refEllipsoid ref_ellipsoid = refEllipsoid::wgs84) {

  double east, north, up;
  aer2enu(az, elev, range, east, north, up);
  enu2uvw(east, north, up, lat0, lon0, x, y, z);
}

/*!
 * @brief 坐标变换：AER ==> Geodetic
 *
 * @param az            目标点方位角，单位 rad
 * @param elev          目标点俯仰角，单位 rad
 * @param range         目标点斜距，单位 m
 * @param lat0          参考点纬度，单位 rad
 * @param lon0          参考点经度，单位 rad
 * @param alt0          参考点高度，单位 m
 * @param[out] lat      目标点纬度，单位 rad
 * @param[out] lon      目标点经度，单位 rad
 * @param[out] alt      目标点高度，单位 m
 * @param               参考椭球模型，'wgs84'（默认） 或 'grs80'
 *
 */
inline void aer2geodetic(double az, double elev, double range, double lat0,
                         double lon0, double alt0, double &lat, double &lon,
                         double &alt,
                         refEllipsoid ref_ellipsoid = refEllipsoid::wgs84) {
  double x, y, z;
  aer2ecef(az, elev, range, lat0, lon0, alt0, x, y, z, ref_ellipsoid);
  ecef2geodetic(x, y, z, lat, lon, alt, ref_ellipsoid);
}

/*!
 * @brief 坐标变换：NED ==> ECEF
 *
 * @param north         目标点北坐标，单位 m
 * @param east          目标点东坐标，单位 m
 * @param down          目标点下坐标，单位 m
 * @param lat0          参考点纬度，单位 rad
 * @param lon0          参考点经度，单位 rad
 * @param alt0          参考点高度，单位 m
 * @param[out] x        目标点 x 坐标，单位 m
 * @param[out] y        目标点 y 坐标，单位 m
 * @param[out] z        目标点 z 坐标，单位 m
 * @param               参考椭球模型，'wgs84'（默认） 或 'grs80'
 *
 */
inline void ned2ecef(double north, double east, double down, double lat0,
                     double lon0, double alt0, double &x, double &y, double &z,
                     refEllipsoid ref_ellipsoid = refEllipsoid::wgs84) {
  enu2ecef(east, north, -1 * down, lat0, lon0, alt0, x, y, z, ref_ellipsoid);
}

/*!
 * @brief 坐标变换：NED ==> Geodetic
 *
 * @param north         目标点北坐标，单位 m
 * @param east          目标点东坐标，单位 m
 * @param down          目标点下，单位 m
 * @param lat0          参考点纬度，单位 rad
 * @param lon0          参考点经度，单位 rad
 * @param alt0          参考点高度，单位 m
 * @param[out] lat      目标点纬度，单位 rad
 * @param[out] lon      目标点经度，单位 rad
 * @param[out] alt      目标点高度，单位 m
 * @param               参考椭球模型，'wgs84'（默认） 或 'grs80'
 *
 */
inline void ned2geodetic(double north, double east, double down, double lat0,
                         double lon0, double alt0, double &lat, double &lon,
                         double &alt,
                         refEllipsoid ref_ellipsoid = refEllipsoid::wgs84) {
  enu2geodetic(east, north, -1 * down, lat0, lon0, alt0, lat, lon, alt,
               ref_ellipsoid);
}

/*!
 * @brief 坐标变换：NED ==> AER
 *
 * @param north         目标点北坐标，单位 m
 * @param east          目标点东坐标，单位 m
 * @param down          目标点下坐标，单位 m
 * @param[out] az       目标点方位角，单位 rad
 * @param[out] elev     目标点俯仰角，单位 rad
 * @param[out] range    目标点斜距，单位 m
 *
 */
inline void ned2aer(double north, double east, double down, double &az,
                    double &elev, double &range) {
  enu2aer(east, north, -1 * down, az, elev, range);
}

/*!
 * @brief  化简经纬度
 *
 * @param lat                 纬度，单位 rad
 * @param lon                 经度，单位 rad
 * @param[out] wrapped_lat    化简后的纬度，单位 rad
 * @param[out] wrapped_lon    化简后的经度，单位 rad
 *
 */

inline void wrapGeodetic(double lat, double lon, double &wrapped_lat,
                         double &wrapped_lon) {

  int quadrant = static_cast<int>(floor(abs(lat) / (M_PI_2))) % 4;
  double pole = (lat > 0) ? (M_PI_2) : -(M_PI_2);
  double offset = std::fmod(lat, (M_PI_2));

  switch (quadrant) {
  case 0:
    wrapped_lat = offset;
    wrapped_lon = lon;
    break;
  case 1:
    wrapped_lat = pole - offset;
    wrapped_lon = lon + M_PI;
    break;
  case 2:
    wrapped_lat = -offset;
    wrapped_lon = lon + M_PI;
    break;
  case 3:
    wrapped_lat = -pole + offset;
    wrapped_lon = lon;
    break;
  }

  if (wrapped_lon > M_PI || wrapped_lon < M_PI) {
    wrapped_lon -= floor((wrapped_lon + M_PI) / (2 * M_PI)) * (2 * M_PI);
  }
}
} // namespace geodesy

#endif
